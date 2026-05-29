# Combo variant of b10624e5 (index 44)
# combo_spec: D3/0+D4/1+D9/0
# Name: 锚块空洞目标 + 矩形锚块 + 非均匀缩放

"""
design_analysis:
  fused_rule:
    keep:
      - 27x27 画布、单条横竖分隔线、四象限布局、TL 单个彩色源组件
      - 右侧象限做左右镜像、下侧象限做上下镜像，BR 同时做两种镜像
    change:
      - 非 TL 目标从“实心红块外贴图案”改成“红色空心矩形框的内部空洞”
      - TL 与缩放基准锚块改为非正方形实心红矩形；其高宽分别编码源单元尺寸
      - 目标缩放不再使用一个统一倍率，而是按目标高宽分别做纵向/横向最近邻缩放
      - 缩放后的整块 patch bbox 必须恰好填满空洞；因此目标槽位和双轴缩放是同一个约束

  concretization_and_trace:
    pixel_form:
      - TL: 一个 4 连通彩色 patch，内部有一个非正方形实心红矩形，例如 2x3
      - TR/BL/BR 输入: 各有一个 1 像素厚红色矩形框，框内全是背景色
      - TR/BL/BR 输出: 红框保留不动，框内被填入对应镜像且按两轴缩放后的完整 patch
    example_trace:
      - 设 TL patch 为 4x6，内部红锚块为 2x3
      - 原题若目标是 4x4 红方块，会把 patch 按同一倍率外贴到红块周围
      - 本融合版若 TR 的隐含目标尺度是 2x1，则先左右镜像，再按高倍率 2/2=1、宽倍率 1/3 缩放，得到 4x2 的 patch
      - 输入里 TR 看到的是一个包住 4x2 空洞的红框；输出时把 4x2 patch 整块塞进空洞，而不是围绕红块展开
      - 差异点:
        - D3: 目标区域从红块外部改成锚块内部空洞
        - D9: 红锚块与目标框都是矩形而非正方形，宽高信息不可压成一个 size
        - D4: 纵横倍率独立，最终图案长宽比必须服从空洞尺寸

  complexity_calibration:
    controls:
      - TL 源锚块只取 2x3 或 3x2 两种，既明确非正方形，又保留足够可视面积
      - 每个样例强制包含三类目标: 只改宽、只改高、两轴都改且倍率不同
      - 红框厚度固定为 1，空洞保持轴对齐矩形，避免 cavity 匹配过于自由
      - TL motif 只用逻辑块上采样生成，确保非均匀缩放可以整除且 roundtrip 精确
    why_not_simpler:
      - 若 TL 锚块仍是 2x2，D9 的“矩形源尺度”会被削弱
      - 若目标框允许任意厚度或不要求矩形，人类很难从 2-3 个样例中稳定位出空洞槽位
      - 若不强制三种拉伸类型，样本可能退化成仅一轴变化或近似等比缩放，难以体现融合

  implicit_assumption_audit:
    original_main_breakages:
      - 原版把红锚块摘要成单个 size，并默认是实心方块；融合后 TL 是实心矩形，非 TL 是空心矩形框，这两个前提都失效
      - 原版把缩放后的红 bbox 对齐到目标红块；融合后目标框只提供空洞，不再提供内部对齐块
      - 原版允许图案落在红锚块周围；融合后必须整个 bbox 落进空洞内部
    new_main_assumptions:
      - TL 象限恰有一个含红色的非背景 4 连通组件，且其红色 bbox 是非正方形实心矩形
      - 非 TL 每个象限恰有一个红色 4 连通 1 像素厚矩形框，内部全是背景
      - 每个空洞尺寸都与“对应镜像后的 TL patch 的某次非均匀缩放”完全一致
      - TL patch 来自按源锚块高宽重复的逻辑网格，因此 resize 到空洞后再缩回原尺寸是无损的
    generator_guarantees:
      - 逐象限只放一个红组件，避免 solver 在红组件匹配时歧义
      - 所有 frame 与 cavity 都完整落在 13x13 象限内
      - 只接受 `main(inp) == out` 的样本，并额外强制非均匀缩放类别分布

  placement_legality:
    - TL patch 完整放入 TL 象限
    - 每个非 TL frame 的外框和空洞都完整放入自己的象限，且互不重叠
    - 输出填充只发生在 frame 内部，不覆盖 divider，也不触碰别的象限

  visual_observability:
    - 输入中三处都是“空心红框 + 空白洞”，输出中洞被彩色 patch 完整填满，D3 非常直观
    - TL 红锚块与输出里的内部红块都是矩形且非正方形，能直观看到 D9
    - 三个目标分别体现横向-only、纵向-only、双轴变化，能从少量样例看出 D4 的双轴缩放
    - 通过拒绝水平/垂直对称的源图案，TR/BL/BR 的镜像关系对人类可见

  neatness_and_inference:
    - 红框始终只扮演“槽位边界”，内部内容与边框分层清楚，不会出现多对象纠缠
    - 不使用颜色 id 做隐藏排序或映射；颜色只服务于图案纹理与镜像可见性
    - 由于 cavity 尺寸、源锚块矩形尺寸、象限镜像三者同时起作用，人类可以从结构变化推断规则，而不是靠枚举颜色或坐标技巧
"""

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
FRAME_THICKNESS = 1
MAX_RETRIES = 500

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SHAPES = [(2, 3), (3, 2)]
LOGICAL_DIMS = [(2, 2), (2, 3), (3, 2)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_RECT_CHOICES = [
    (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 4),
    (4, 1), (4, 2), (4, 3),
]


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
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
    """Nearest-neighbor resize with independent row/column targets."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resized patch must stay positive"
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


def _is_filled_rect(red_patch: np.ndarray) -> bool:
    return bool(red_patch.size > 0 and np.all(red_patch == RED))


def _is_rect_frame(red_patch: np.ndarray, bg: int) -> bool:
    h, w = red_patch.shape
    if h < 3 or w < 3:
        return False
    if not np.all(red_patch[0, :] == RED):
        return False
    if not np.all(red_patch[-1, :] == RED):
        return False
    if not np.all(red_patch[:, 0] == RED):
        return False
    if not np.all(red_patch[:, -1] == RED):
        return False
    return bool(np.all(red_patch[1:-1, 1:-1] == bg))


def _red_bbox_dims(img: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "patch must contain red pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    h = int(rr.max() - rr.min() + 1)
    w = int(cc.max() - cc.min() + 1)
    return r0, c0, h, w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the hollow rectangular targets with mirrored, non-uniformly scaled copies of the top-left motif.

    Input structure:
    - One full divider row and one full divider column split the 27x27 board into four 13x13 quadrants.
    - The top-left quadrant contains a single colored 4-connected motif whose red anchor is a filled non-square rectangle.
    - Each other quadrant contains exactly one hollow red rectangular frame, and the frame interior is empty background.

    Rule:
    - Extract the TL motif component as the source patch.
    - Reflect it left/right for right quadrants and up/down for bottom quadrants.
    - Resize the whole reflected patch so its height and width exactly equal the target frame's interior cavity, then paste that resized patch into the cavity.

    Key insight:
    - The target red object no longer tells you where to attach the motif externally. Its empty rectangular cavity is the destination, and that cavity's height and width jointly encode the required non-uniform stretch.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    mid_r, mid_c = _find_dividers(g, bg)

    red_comps = components_mask(g == RED, connectivity=4)
    assert red_comps, "grid must contain red components"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red component"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one red component per quadrant"

    sr0, sc0, sr1, sc1 = anchors[(0, 0)]
    src_red = g[sr0:sr1 + 1, sc0:sc1 + 1]
    src_h, src_w = src_red.shape
    assert src_h != src_w, "top-left red anchor must be a non-square rectangle"
    assert _is_filled_rect(src_red), "top-left red anchor must be filled"

    tl = g[0:mid_r, 0:mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with red"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0:pr1 + 1, pc0:pc1 + 1].copy()
    red_r0, red_c0, red_h, red_w = _red_bbox_dims(patch)
    assert (red_h, red_w) == (src_h, src_w), "source patch red bbox must match the TL anchor rectangle"
    assert src_h <= patch.shape[0] and src_w <= patch.shape[1]

    for (qb, qr), (r0, c0, r1, c1) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        red_patch = g[r0:r1 + 1, c0:c1 + 1]
        assert _is_rect_frame(red_patch, bg), "non-TL red component must be a hollow rectangular frame"
        assert red_patch.shape[0] != red_patch.shape[1], "non-TL target frames must be non-square"

        cavity_h = int(red_patch.shape[0] - 2 * FRAME_THICKNESS)
        cavity_w = int(red_patch.shape[1] - 2 * FRAME_THICKNESS)
        patch_t = _reflect_patch(patch, qb, qr)
        patch_s = _resize_nn(patch_t, cavity_h, cavity_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "cavity-driven anisotropic scaling must preserve the motif"
        _, _, tgt_h, tgt_w = _red_bbox_dims(patch_s)
        assert tgt_h != tgt_w, "scaled internal red anchor must stay rectangular"
        assert cavity_h * src_h == patch_t.shape[0] * tgt_h, "cavity height must match source-height scaling"
        assert cavity_w * src_w == patch_t.shape[1] * tgt_w, "cavity width must match source-width scaling"

        paste(out, patch_s, r0 + FRAME_THICKNESS, c0 + FRAME_THICKNESS)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
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
    extras = random.sample(COLOR_POOL, n_colors - 1)
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _logical_patch_is_observable(logical: np.ndarray, ar: int, ac: int) -> bool:
    non_bg = logical != BG
    same_row = any(non_bg[ar, c] and logical[ar, c] != RED for c in range(logical.shape[1]))
    same_col = any(non_bg[r, ac] and logical[r, ac] != RED for r in range(logical.shape[0]))
    asym_h = not np.array_equal(logical, np.fliplr(logical))
    asym_v = not np.array_equal(logical, np.flipud(logical))
    return same_row and same_col and asym_h and asym_v


def _source_patch(source_shape: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif on a logical grid and upscale each logical cell by the source anchor shape."""
    ah, aw = source_shape
    lh, lw = random.choice(LOGICAL_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
    if logical is None or not _logical_patch_is_observable(logical, ar, ac):
        return None
    patch = np.repeat(np.repeat(logical, ah, axis=0), aw, axis=1)
    return patch, ar * ah, ac * aw


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    p = _reflect_patch(patch, qb, qr)
    sh, sw = source_shape
    th, tw = target_shape
    out_h = max(1, int(round(p.shape[0] * th / sh)))
    out_w = max(1, int(round(p.shape[1] * tw / sw)))
    return _resize_nn(p, out_h, out_w)


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


def _fits_quadrant(patch: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> bool:
    sh, sw = source_shape
    th, tw = target_shape
    cavity_h = int(round(patch.shape[0] * th / sh))
    cavity_w = int(round(patch.shape[1] * tw / sw))
    outer_h = cavity_h + 2 * FRAME_THICKNESS
    outer_w = cavity_w + 2 * FRAME_THICKNESS
    return outer_h <= 13 and outer_w <= 13 and outer_h != outer_w


def _is_non_uniform(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> bool:
    sh, sw = source_shape
    th, tw = target_shape
    return th * sw != tw * sh


def _choose_target_shapes(
    source_shape: tuple[int, int],
    patch: np.ndarray,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    sh, sw = source_shape
    feasible = [rc for rc in TARGET_RECT_CHOICES if _fits_quadrant(patch, source_shape, rc)]
    horiz_only = [rc for rc in feasible if rc[0] == sh and rc[1] != sw]
    vert_only = [rc for rc in feasible if rc[1] == sw and rc[0] != sh]
    full_aniso = [rc for rc in feasible if rc[0] != sh and rc[1] != sw and _is_non_uniform(source_shape, rc)]
    if not horiz_only or not vert_only or not full_aniso:
        return None

    for _ in range(40):
        picks = [random.choice(horiz_only), random.choice(vert_only), random.choice(full_aniso)]
        random.shuffle(picks)
        keys = [(0, 1), (1, 0), (1, 1)]
        scaled_sizes = []
        for rc in picks:
            cavity_h = int(round(patch.shape[0] * rc[0] / sh))
            cavity_w = int(round(patch.shape[1] * rc[1] / sw))
            scaled_sizes.append((cavity_h, cavity_w))
        if len(set(scaled_sizes)) < 3:
            continue
        if not all(h != w for h, w in scaled_sizes):
            continue
        return {(0, 0): source_shape, **{k: v for k, v in zip(keys, picks, strict=False)}}
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    source_shape = random.choice(SOURCE_ANCHOR_SHAPES)
    source = _source_patch(source_shape)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    target_shapes = _choose_target_shapes(source_shape, patch)
    if target_shapes is None:
        return None

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], source_shape, target_shapes[key])
        for key in target_shapes
        if key != (0, 0)
    }
    for key, patch_s in scaled.items():
        _, _, red_h, red_w = _red_bbox_dims(patch_s)
        if (red_h, red_w) != target_shapes[key]:
            return None
        if patch_s.shape[0] == patch_s.shape[1]:
            return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    tl_r0, tl_c0 = _choose_quad_position((0, 0), patch.shape[0], patch.shape[1])
    paste(out, patch, tl_r0, tl_c0)

    frame_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for key, patch_s in scaled.items():
        frame = _frame_patch(*patch_s.shape)
        if frame.shape[0] == frame.shape[1]:
            return None
        fr0, fc0 = _choose_quad_position(key, frame.shape[0], frame.shape[1])
        paste(out, frame, fr0, fc0)
        paste(out, patch_s, fr0 + FRAME_THICKNESS, fc0 + FRAME_THICKNESS)
        frame_boxes[key] = (fr0, fc0, frame.shape[0], frame.shape[1])

    inp = out.copy()
    for key, (fr0, fc0, fh, fw) in frame_boxes.items():
        inp[fr0 + FRAME_THICKNESS:fr0 + fh - FRAME_THICKNESS, fc0 + FRAME_THICKNESS:fc0 + fw - FRAME_THICKNESS] = BG

    if inp[tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch] != RED:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
