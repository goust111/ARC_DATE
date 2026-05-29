# Variant of b10624e5
# Dimension: D3 — 目标对象：其他象限的红色锚块定位 (dst)
# Change: 其他象限中红色锚块的相邻位置作为目标，图案根据锚块尺寸和象限关系进行缩放和反射后放置... → 锚块内部的空洞作为目标槽位

"""
design_analysis:
  variant_goal:
    keep:
      - 四象限布局、十字分隔线、TL 源图案、按象限做左右/上下镜像
      - 缩放仍然由 TL 的 2x2 红锚块作为基准，目标尺度仍取 1..4 档
    change:
      - 原题把缩放后的图案对齐到目标红方块并向外盖章
      - 本变体把非 TL 目标改成 1 像素厚的红色空心框，内部矩形空洞是唯一合法槽位
      - 输出时不再向锚块外部扩展，而是把缩放+镜像后的整块 TL patch 精确塞进空洞 bbox

  concretization_and_trace:
    source_object:
      - TL 仍保留一个包含 2x2 红锚块的单个 4 连通彩色 patch
      - patch bbox 是后续缩放的整体对象；其中红色子块只负责提供尺度基准
    destination_object:
      - 其他三个象限各有一个红色矩形框，框内全是背景色
      - 框内空洞尺寸 == 对应缩放后 patch 的尺寸，因此空洞本身就是 dst
    example_trace:
      - 原逻辑: TR 有 3x3 实心红块，先把 TL patch 左右翻转，再按 3/2 缩放，然后让 patch 中红块覆盖目标红块，剩余像素落在其周围
      - 新逻辑: TR 改成包住一个 6x5 空洞的红框，仍做左右翻转和 3/2 缩放，但现在把 6x5 整个 patch bbox 直接放入空洞左上角；红框只负责提供槽位，不被覆盖
      - 差异: “目标定位点” 从红块相邻区域变成红框内部的矩形洞槽；图案落点从外贴变成内填

  complexity_calibration:
    choices:
      - 空洞保持矩形且四边框厚度固定为 1，避免 cavity 过自由导致匹配分支过多
      - 仍保留 1..4 四档尺度，但只在洞槽能完整容纳时使用
      - 要求 TL patch 对左右/上下翻转至少都不是完全对称，保证反射在人眼上可见
    why_not_simpler:
      - 如果直接把目标改成任意空白矩形而没有红框，人类很难知道哪些空白区是槽位
      - 如果所有目标尺寸都固定或都等于 2，则 D3 的“洞槽约束”存在，但缩放信息会变弱

  implicit_assumption_audit:
    original_main_lines_affected:
      - 原版通过 red 组件 bbox 直接得到目标锚块大小，并把 scaled red bbox 对齐到该方块
      - 该写法隐含前提: 目标红物体是实心方块，而且图案可以落在方块周边
      - 变体后此假设失效，因为非 TL 红物体变成空心框；若仍按原逻辑对齐 red bbox，会把图案错误贴到框外
    new_solver_assumptions:
      - 非 TL 每个象限恰有一个红色 4 连通矩形框
      - 红框内部全是背景，且空洞 bbox 与某个 1..4 尺度的缩放 patch 尺寸唯一匹配
      - TL patch 所在连通块仍是该象限唯一含红色的非背景连通块
    generator_guarantees:
      - 每个非 TL 红框都严格是 1 像素厚矩形边框
      - 只选择会让 scaled patch 完整放进 13x13 象限的尺度
      - 若可行尺度多于 1 个，尽量采样至少两种不同尺度，避免样本退化

  placement_legality:
    - TL patch 完整放在 TL 象限内
    - 每个目标红框完整放在自己的象限内，且框内空洞完整放在象限内
    - 输出时 patch 只写入洞槽内部，不碰分隔线、不越界

  visual_observability:
    - 输入中目标框内是干净空洞，输出中该空洞被完整填入彩色图案，D3 变化一眼可见
    - 通过要求 TL patch 对 LR / UD 都不完全对称，TR / BL / BR 三象限的结果可体现镜像差异
    - 不依赖颜色 id 做排序或变换；颜色只作为图案本身的可见纹理

  neatness:
    - 一个象限只放一个目标框，避免多对象重叠
    - 红框与内部内容分层清楚：边框始终保留，内容始终在框内
"""

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _is_rect_frame(red_patch: np.ndarray, bg: int) -> bool:
    h, w = red_patch.shape
    if h < 3 or w < 3:
        return False
    if not np.all(red_patch[0, :] == 2):
        return False
    if not np.all(red_patch[-1, :] == 2):
        return False
    if not np.all(red_patch[:, 0] == 2):
        return False
    if not np.all(red_patch[:, -1] == 2):
        return False
    return bool(np.all(red_patch[1:-1, 1:-1] == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif into hollow red target frames using scaling and reflection.

    Input structure:
    - A full-row/full-col divider splits the board into four quadrants.
    - The top-left quadrant contains one colored 4-connected motif with a filled 2x2 red anchor inside it.
    - Each other quadrant contains one hollow red rectangular frame whose interior is empty background.

    Rule:
    - Extract the TL motif component and use it as the source patch.
    - For every non-TL quadrant, reflect the source according to quadrant position and scale it by one of the anchor-size ratios 1/2, 1, 3/2, or 2.
    - Choose the unique scale whose resized patch dimensions exactly match the target frame's interior cavity, then place the whole resized patch into that cavity.

    Key insight: The red target no longer marks an external attachment point; its empty interior is the destination slot, and the slot size reveals which scaled mirror of the TL patch must be inserted.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    mid_r, mid_c = _find_dividers(g, bg)

    red_comps = components_mask(g == 2, connectivity=4)
    assert red_comps, "grid must contain red components"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red component"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need exactly one red component per quadrant"

    sr0, sc0, sr1, sc1 = anchors[(0, 0)]
    sL = max(sr1 - sr0 + 1, sc1 - sc0 + 1)
    assert sL > 0, "top-left red anchor must have positive size"

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == 2 for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (r0, c0, r1, c1) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        red_patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_rect_frame(red_patch, bg), "non-TL red target must be a hollow rectangular frame"
        cavity_h = red_patch.shape[0] - 2
        cavity_w = red_patch.shape[1] - 2

        patch_t = _reflect_patch(patch, qb, qr)
        matches: list[np.ndarray] = []
        for target_size in (1, 2, 3, 4):
            ratio = float(target_size) / float(sL)
            new_h = max(1, int(round(patch_t.shape[0] * ratio)))
            new_w = max(1, int(round(patch_t.shape[1] * ratio)))
            if (new_h, new_w) != (cavity_h, cavity_w):
                continue
            patch_s = _resize_nn(patch_t, new_h, new_w)
            roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
            assert np.array_equal(roundtrip, patch_t), "scaling must preserve motif information"
            matches.append(patch_s)

        assert len(matches) == 1, "target cavity must match exactly one scaled reflected source patch"
        paste(out, matches[0], r0 + 1, c0 + 1)

    return out


# --------------- generator ---------------

BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
SOURCE_ANCHOR_SIZE = 2
FRAME_THICKNESS = 1
MAX_CAVITY_DIM = 11
TARGET_SIZE_CHOICES = (1, 2, 3, 4)
MAX_RETRIES = 400

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.remove(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None

    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    s = SOURCE_ANCHOR_SIZE
    use_block = random.random() < BLOCK_PROB
    if use_block:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar = random.randint(0, h - s)
    ac = random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _feasible_sizes(patch: np.ndarray) -> list[int]:
    ok = []
    for target_size in TARGET_SIZE_CHOICES:
        ratio = target_size / SOURCE_ANCHOR_SIZE
        new_h = max(1, int(round(patch.shape[0] * ratio)))
        new_w = max(1, int(round(patch.shape[1] * ratio)))
        if new_h > MAX_CAVITY_DIM or new_w > MAX_CAVITY_DIM:
            continue
        patch_s = _resize_nn(patch, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch.shape[0], patch.shape[1])
        if np.array_equal(roundtrip, patch):
            ok.append(int(target_size))
    return ok


def _frame_patch(cavity_h: int, cavity_w: int) -> np.ndarray:
    frame = np.full((cavity_h + 2 * FRAME_THICKNESS, cavity_w + 2 * FRAME_THICKNESS), RED, dtype=int)
    frame[FRAME_THICKNESS:-FRAME_THICKNESS, FRAME_THICKNESS:-FRAME_THICKNESS] = BG
    return frame


def _choose_quad_position(key: tuple[int, int], patch_h: int, patch_w: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    max_r0 = q_r1 - patch_h + 1
    max_c0 = q_c1 - patch_w + 1
    assert q_r0 <= max_r0 and q_c0 <= max_c0
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _visibly_asymmetric(patch: np.ndarray) -> bool:
    return (not np.array_equal(patch, np.fliplr(patch))) and (not np.array_equal(patch, np.flipud(patch)))


def _choose_target_sizes(feasible: list[int]) -> dict[tuple[int, int], int]:
    keys = [(0, 1), (1, 0), (1, 1)]
    for _ in range(40):
        picks = [random.choice(feasible) for _ in keys]
        if len(feasible) >= 2 and len(set(picks)) < 2:
            continue
        if any(v != SOURCE_ANCHOR_SIZE for v in picks) or len(feasible) == 1:
            return dict(zip(keys, picks, strict=False))
    picks = [random.choice(feasible) for _ in keys]
    return dict(zip(keys, picks, strict=False))


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    if not _visibly_asymmetric(patch):
        return None

    feasible = _feasible_sizes(patch)
    if not feasible:
        return None

    target_sizes = _choose_target_sizes(feasible)
    scaled = {
        key: _resize_nn(
            _reflect_patch(patch, key[0], key[1]),
            max(1, int(round(patch.shape[0] * target_sizes[key] / SOURCE_ANCHOR_SIZE))),
            max(1, int(round(patch.shape[1] * target_sizes[key] / SOURCE_ANCHOR_SIZE))),
        )
        for key in target_sizes
    }

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    tl_r0, tl_c0 = _choose_quad_position((0, 0), patch.shape[0], patch.shape[1])
    paste(out, patch, tl_r0, tl_c0)

    frame_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for key, patch_s in scaled.items():
        frame = _frame_patch(*patch_s.shape)
        fr0, fc0 = _choose_quad_position(key, frame.shape[0], frame.shape[1])
        paste(out, frame, fr0, fc0)
        paste(out, patch_s, fr0 + FRAME_THICKNESS, fc0 + FRAME_THICKNESS)
        frame_boxes[key] = (fr0, fc0, frame.shape[0], frame.shape[1])

    inp = out.copy()
    for key, (fr0, fc0, fh, fw) in frame_boxes.items():
        inp[fr0 + FRAME_THICKNESS : fr0 + fh - FRAME_THICKNESS, fc0 + FRAME_THICKNESS : fc0 + fw - FRAME_THICKNESS] = BG

    if inp[tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch] != RED:
        return None
    if np.array_equal(main(inp), out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid b10624e5 D3 input")
