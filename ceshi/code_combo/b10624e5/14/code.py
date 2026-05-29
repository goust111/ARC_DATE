# Combo variant of b10624e5 (index 14)
# combo_spec: D4/0+D9/2
# Name: 所有锚块尺寸相同 + 固定缩放因子
#
# ---------------------------------------------------------------------------
# 设计分析（融合版：D4/0 + D9/2）
# ---------------------------------------------------------------------------
# a) 具体化与行为追踪
# - 原题：四个象限各有一个红色方锚。左上象限给出完整图案，其他象限只给红锚。复制时把整块
#   图案按“目标锚尺寸 / 源锚尺寸”做最近邻缩放，再按象限关系做左右/上下反射。
# - D9/2 去掉了“锚尺寸不同”这条线索；D4/0 又要求仍然存在统一固定倍率。两者叠加后，若还把
#   红锚本身一起等比放大，就无法和“所有锚块尺寸相同”共存。
# - 因此这里把规则具体化为：先把含红锚的整块 patch 反射并按固定 2 倍放大；随后把放大后的红锚
#   从 4x4 缩回原来的 2x2，只保留靠近彩色图案的那个“内角”子块。这样输入里四个锚全都同尺寸，
#   但目标图案的彩色部分仍明显比左上源图案更大。
# - 小例子：若左上 patch 的红锚在 patch 左下角，彩色像素位于锚的上方和右方，那么右上象限先做
#   左右反射得到“右下角红锚、彩色像素在上方和左方”的 patch；再整体 2 倍放大；最后把 4x4 红锚
#   缩回为左上角那块 2x2，使放大后的彩色部分仍紧贴红锚。这和原题的“尺寸比=1 所以只镜像不放大”
#   有明确差异。
#
# b) 复杂度校准
# - 固定倍率若同时允许 2 倍和 3 倍，会让少量样例难以归纳，因此整个任务集固定为 2 倍。
# - 若源图案太小或太靠近锚，2 倍后和 1:1 反射的差别不够显眼。生成器强制源 patch 在锚的上方和
#   右方都至少延伸两格，并要求输出相对“只反射不放大”的旧逻辑至少有若干像素差异。
# - 若源图案本身水平/垂直完全对称，反射线索会变弱。生成器显式拒绝这类弱样本。
#
# c) 原 main() 的隐式假设审查
# - 原 `ratio = tL / sL` 依赖“不同象限锚尺寸可能不同”；融合后这个前提失效，必须删掉动态比例，
#   改成常量 `SCALE_FACTOR = 2`。
# - 原 `patch_s == 2` 默认放大后的红锚尺寸就等于目标锚尺寸；融合后不成立，所以 solver 需要多一层
#   “放大后再把红锚缩回原尺寸”的处理。
# - 原逻辑只要求 TL 连通块里含红色即可；这里新增前提是：TL 源 patch 的红锚必须贴住该 patch 的一
#   个角，彩色部分位于锚的“内侧”。否则“缩回哪一个红锚子角”没有唯一答案。生成器会严格构造并保
#   证这一点。
#
# d) 放置合法性
# - 四个象限仍由整行整列分割线分开，所有对象都完全放在各自 13x13 象限内，不跨越分割线。
# - 生成器先根据 2 倍变体的最终尺寸计算每个目标 patch，再反推锚块可放置范围，因此不会越界。
# - 每个象限只有一个红锚；非左上象限输入里除了红锚没有其他前景，避免配对歧义。
#
# e) 视觉可观察性
# - 解题者在 2-4 个样例里能稳定看到两件事：四个红锚总是一样大；右/下象限的彩色图案总是“镜像后
#   再统一变大一档”，而不是随锚尺寸变化。
# - 颜色不承担隐藏编码功能，只用于显示局部形状。固定倍率通过几何尺寸变化体现，不依赖颜色数值。
#
# f) 可视化整洁度
# - TL 源 patch 一律是“角锚 + 朝内生长”的单个连通块，布局干净。
# - 目标 patch 也都完整落在象限内，避免大量裁剪造成视觉噪声。
# - 不使用分割线颜色作为图案颜色，保证分区结构和图案结构清晰可分。
#
# g) main() 的新隐式假设与 generator 的对应保证
# - 假设 1：每个象限恰有一个红色方形锚。generator 固定如此构造。
# - 假设 2：四个锚大小完全一致。generator 统一使用 2x2 红锚。
# - 假设 3：TL 源 patch 的红锚恰贴 patch 的一个角。generator 固定把它放在左下角，并让图案向内侧
#   延伸；solver 反射后再根据反射结果自动识别应保留哪一个“内角”红块。
# - 假设 4：固定 2 倍缩放不能退化成原题的 1:1 反射。generator 用量化指标检查 fused 输出与
#   old-style 输出的差异像素数，差异不足则重采样。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS = ROOT / "solvers"
if str(SOLVERS) not in sys.path:
    sys.path.insert(0, str(SOLVERS))

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
ANCHOR_SIZE = 2
SCALE_FACTOR = 2
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
COLOR_POOL = [3, 5, 6, 7, 8, 9]
UP_EXTENT_RANGE = (2, 5)
RIGHT_EXTENT_RANGE = (2, 5)
FILL_RATIO_RANGE = (0.45, 0.7)
EXTRA_COLORS_RANGE = (2, 4)
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the matching full-row/full-col divider coordinates."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches: list[tuple[int, int]] = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching divider row/col"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_bbox(patch: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(patch == RED)
    assert rr.size > 0, "patch must contain a red anchor"
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _corner_mode_from_patch(patch: np.ndarray, anchor_size: int) -> tuple[str, str]:
    """Return which corner of the scaled red block should be kept after fixed scaling."""
    h, w = patch.shape
    r0, c0, r1, c1 = _anchor_bbox(patch)
    assert r1 - r0 + 1 == anchor_size and c1 - c0 + 1 == anchor_size, "anchor must keep its base size before scaling"
    if r1 == h - 1 and c0 == 0:
        return "top", "right"
    if r1 == h - 1 and c1 == w - 1:
        return "top", "left"
    if r0 == 0 and c0 == 0:
        return "bottom", "right"
    if r0 == 0 and c1 == w - 1:
        return "bottom", "left"
    raise AssertionError("source patch anchor must sit in a patch corner")


def _shrink_scaled_anchor(
    patch_s: np.ndarray, anchor_size: int, keep_rows: str, keep_cols: str
) -> np.ndarray:
    """Shrink the scaled red anchor back to its original size while keeping the inner corner."""
    r0, c0, r1, c1 = _anchor_bbox(patch_s)
    red_h = r1 - r0 + 1
    red_w = c1 - c0 + 1
    assert red_h >= anchor_size and red_w >= anchor_size, "scaled anchor must be at least as large as the base anchor"

    row_mask = np.ones(patch_s.shape[0], dtype=bool)
    col_mask = np.ones(patch_s.shape[1], dtype=bool)

    if keep_rows == "top":
        row_mask[r0 + anchor_size : r1 + 1] = False
    else:
        row_mask[r0 : r1 - anchor_size + 1] = False

    if keep_cols == "left":
        col_mask[c0 + anchor_size : c1 + 1] = False
    else:
        col_mask[c0 : c1 - anchor_size + 1] = False

    shrunk = patch_s[row_mask][:, col_mask]
    sr0, sc0, sr1, sc1 = _anchor_bbox(shrunk)
    assert sr1 - sr0 + 1 == anchor_size and sc1 - sc0 + 1 == anchor_size, "shrunk anchor must recover the base size"
    return shrunk


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _fixed_scale_patch(patch: np.ndarray, qb: int, qr: int, anchor_size: int) -> np.ndarray:
    """
    Reflect the source patch into the target quadrant, enlarge it by a fixed factor,
    then shrink the red anchor back to the common anchor size.
    """
    reflected = _reflect_patch(patch, qb, qr)
    keep_rows, keep_cols = _corner_mode_from_patch(reflected, anchor_size)
    scaled = _resize_nn(reflected, reflected.shape[0] * SCALE_FACTOR, reflected.shape[1] * SCALE_FACTOR)
    shrunk = _shrink_scaled_anchor(scaled, anchor_size, keep_rows, keep_cols)
    return shrunk


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    red_comps = components_mask(g == RED, connectivity=4)
    assert len(red_comps) == 4, "grid must contain exactly four red anchors"
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one anchor"
        assert r1 - r0 + 1 == size and c1 - c0 + 1 == size, "anchors must be square"
        anchors[key] = (int(r0), int(c0), int(size))
    sizes = {info[2] for info in anchors.values()}
    assert len(sizes) == 1, "all anchors must share one common size"
    return anchors


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int, anchor_size: int) -> np.ndarray:
    tl = g[:mid_r, :mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain the source component"
    r0, c0, r1, c1 = patch_bb
    patch = tl[r0 : r1 + 1, c0 : c1 + 1].copy()
    _corner_mode_from_patch(patch, anchor_size)
    return patch


def _stamp_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int, mid_r: int, mid_c: int) -> None:
    rr, cc = np.where(patch != bg)
    for pr, pc in zip(rr.tolist(), cc.tolist(), strict=False):
        tr = top + pr
        tc = left + pc
        if not (0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]):
            continue
        if tr == mid_r or tc == mid_c:
            continue
        out[tr, tc] = int(patch[pr, pc])


def _place_from_anchor(
    out: np.ndarray,
    patch: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    rr, cc = np.where(patch == RED)
    top = anchor_r - int(rr.min())
    left = anchor_c - int(cc.min())
    _stamp_patch(out, patch, top, left, bg, mid_r, mid_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif to the other quadrants using reflection plus one fixed enlargement,
    while keeping all red anchors the same size.

    Input structure:
    - A single non-background divider row and divider column split the board into four 13x13 quadrants.
    - Each quadrant contains exactly one red square anchor, and all four anchors have the same size.
    - The top-left quadrant contains one connected colored source patch whose red anchor sits in a patch corner.

    Rule:
    - Take the top-left source patch as the template.
    - For each other quadrant, reflect the patch according to quadrant position:
      right quadrants flip left-right, bottom quadrants flip up-down, bottom-right does both.
    - Enlarge the reflected patch by a fixed factor of 2, then shrink only the red anchor back to the common
      anchor size by keeping the inner corner of the enlarged red block.
    - Align that restored red anchor to the target quadrant's red anchor and stamp the transformed patch.

    Key insight:
    - Anchor size no longer tells you the scale; all anchors match.
    - The constant 2x growth must be inferred from examples, while quadrant position still determines the reflections.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)

    anchor_size = next(iter(info[2] for info in anchors.values()))
    patch = _extract_source_patch(g, bg, mid_r, mid_c, anchor_size)

    for (qb, qr), (ar, ac, _size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        transformed = _fixed_scale_patch(patch, qb, qr, anchor_size)
        _place_from_anchor(out, transformed, ar, ac, bg, mid_r, mid_c)

    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _grow_4conn(
    h: int,
    w: int,
    seeds: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    n_fill: int,
) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier: set[tuple[int, int]] = set()
    for r, c in filled:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled and (nr, nc) not in blocked:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        filled.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled and (nr, nc) not in blocked:
                frontier.add((nr, nc))
    return filled


def _assign_patch_colors(
    h: int,
    w: int,
    filled: set[tuple[int, int]],
    anchor_cells: set[tuple[int, int]],
) -> np.ndarray | None:
    if not filled:
        return None
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if 0 not in rows or (w - 1) not in cols:
        return None

    patch = np.full((h, w), BG, dtype=int)
    for r, c in anchor_cells:
        patch[r, c] = RED

    cells = list(filled)
    random.shuffle(cells)
    max_extra = min(EXTRA_COLORS_RANGE[1], len(cells))
    if max_extra < EXTRA_COLORS_RANGE[0]:
        return None
    n_extra = random.randint(EXTRA_COLORS_RANGE[0], max_extra)
    extras = random.sample(COLOR_POOL, n_extra)
    for i, (r, c) in enumerate(cells):
        patch[r, c] = extras[i] if i < n_extra else random.choice(extras)
    return patch


def _make_source_patch() -> np.ndarray | None:
    up = random.randint(*UP_EXTENT_RANGE)
    right = random.randint(*RIGHT_EXTENT_RANGE)
    h = up + ANCHOR_SIZE
    w = ANCHOR_SIZE + right

    anchor_cells = {(r, c) for r in range(up, h) for c in range(ANCHOR_SIZE)}
    seed_choices = {(up - 1, 0), (up, ANCHOR_SIZE)}
    n_accessible = h * w - len(anchor_cells)
    fill_min = max(len(seed_choices) + 3, int(np.ceil(FILL_RATIO_RANGE[0] * n_accessible)))
    fill_max = max(fill_min, int(np.floor(FILL_RATIO_RANGE[1] * n_accessible)))
    filled = _grow_4conn(h, w, seed_choices, anchor_cells, random.randint(fill_min, fill_max))

    patch = _assign_patch_colors(h, w, filled, anchor_cells)
    if patch is None:
        return None
    if np.array_equal(patch, np.fliplr(patch)) or np.array_equal(patch, np.flipud(patch)):
        return None

    try:
        keep_rows, keep_cols = _corner_mode_from_patch(patch, ANCHOR_SIZE)
    except AssertionError:
        return None
    if keep_rows != "top" or keep_cols != "right":
        return None
    return patch


def _choose_anchor(
    qb: int,
    qr: int,
    patch: np.ndarray,
    anchor_size: int,
) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch == RED)
    red_r0 = int(rr.min())
    red_c0 = int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch.shape[1] + 1 + red_c0

    anchor_r0 = q_r0
    anchor_r1 = q_r1 - anchor_size + 1
    anchor_c0 = q_c0
    anchor_c1 = q_c1 - anchor_size + 1

    r_lo = max(fit_r0, anchor_r0)
    r_hi = min(fit_r1, anchor_r1)
    c_lo = max(fit_c0, anchor_c0)
    c_hi = min(fit_c1, anchor_c1)
    assert r_lo <= r_hi and c_lo <= c_hi, "target patch must fit fully inside its quadrant"
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _build_output(
    source_patch: np.ndarray,
    tl_top: int,
    tl_left: int,
    anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    paste(out, source_patch, tl_top, tl_left)
    for ar, ac in anchors.values():
        paste(out, np.full((ANCHOR_SIZE, ANCHOR_SIZE), RED, dtype=int), ar, ac)
    for key, (ar, ac) in anchors.items():
        if key == (0, 0):
            continue
        patch = _fixed_scale_patch(source_patch, key[0], key[1], ANCHOR_SIZE)
        _place_from_anchor(out, patch, ar, ac, BG, MID, MID)
    return out


def _build_input(
    source_patch: np.ndarray,
    tl_top: int,
    tl_left: int,
    anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, source_patch, tl_top, tl_left)
    for ar, ac in anchors.values():
        paste(inp, np.full((ANCHOR_SIZE, ANCHOR_SIZE), RED, dtype=int), ar, ac)
    return inp


def _legacy_unscaled_output(inp: np.ndarray) -> np.ndarray:
    """Original-like behavior after D9/2: reflect only, with no scaling."""
    g = np.asarray(inp, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)
    patch = _extract_source_patch(g, bg, mid_r, mid_c, ANCHOR_SIZE)
    for (qb, qr), (ar, ac, _size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        reflected = _reflect_patch(patch, qb, qr)
        _place_from_anchor(out, reflected, ar, ac, bg, mid_r, mid_c)
    return out


def _difference_metric(inp: np.ndarray) -> int:
    fused = main(inp)
    legacy = _legacy_unscaled_output(inp)
    return int(np.count_nonzero(fused != legacy))


def _gen_attempt() -> np.ndarray | None:
    source_patch = _make_source_patch()
    if source_patch is None:
        return None

    tl_top = random.randint(0, MID - source_patch.shape[0])
    tl_left = random.randint(0, MID - source_patch.shape[1])
    tl_anchor_row = tl_top + source_patch.shape[0] - ANCHOR_SIZE
    tl_anchor_col = tl_left

    transformed = {
        key: _fixed_scale_patch(source_patch, key[0], key[1], ANCHOR_SIZE)
        for key in QUADS
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {(0, 0): (tl_anchor_row, tl_anchor_col)}
    for key, patch in transformed.items():
        anchors[key] = _choose_anchor(key[0], key[1], patch, ANCHOR_SIZE)

    out = _build_output(source_patch, tl_top, tl_left, anchors)
    inp = _build_input(source_patch, tl_top, tl_left, anchors)

    if not np.array_equal(main(inp), out):
        return None
    if _difference_metric(inp) < 10:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
