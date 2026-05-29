# Combo variant of b10624e5 (index 3)
# combo_spec: D2/2+D3/0
# Name: 锚块内部嵌入源图案 + 锚块空洞目标槽位

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
design_analysis:
  concretization_and_trace:
    keep:
      - 27x27 网格、十字分隔线、四象限结构、按象限做 LR/UD 反射
      - 缩放仍只使用少量离散档位，便于从少数样例归纳
    fused_change:
      - TL 不再有锚块外侧的 motif；改为一个 4x4 红色方框，内部 2x2 彩色微图案就是唯一源
      - 非 TL 不再是实心红块；改为 1 像素厚红色空心方框，内部背景空洞是唯一目标槽位
      - 输出流程变成：从 TL 红框内部提取 2x2 微图案 -> 按象限镜像 -> 按目标空洞边长缩放 -> 填进对应空洞
    small_trace:
      - 例：TL 内部是 [[a,b],[c,d]]
      - 原题 TR: 取锚块外整块 patch，左右翻转后围绕目标红块向外盖章
      - 融合版 TR: 只取 TL 内部 2x2，先左右翻成 [[b,a],[d,c]]，若 TR 空洞边长为 3，则缩成 3x3 后整块塞进红框内部
      - 差异: 源提取点从“锚块外连通块”变成“锚块内嵌微图案”；目标落点从“锚块外贴合”变成“锚块内填充”

  complexity_calibration:
    choices:
      - TL 内部固定 2x2，避免“从锚块内部提取什么范围”出现歧义
      - 目标空洞边长只取 1..4 四档，既能体现缩放，也不会让槽位尺寸过多
      - 强制 TL 微图案对 LR/UD 都非对称，让三处目标的镜像差异肉眼可见
      - 三个目标中至少两个尺寸不同，且至少一个尺寸不等于 2，保证尺寸变量真实起作用
    why_this_is_not_too_simple:
      - 若所有空洞都同尺寸，解题者可能只看到“复制到框里”而忽略缩放
      - 若 TL 内部允许更大自由形状，人类要先猜边界再猜缩放，信息噪声过高

  implicit_assumption_audit:
    original_lines_that_break:
      - 原版把 TL 含红连通块整体当作 source patch；这里该假设失效，因为真正信息只在红框内部
      - 原版把 scaled patch 的红区域对齐到目标红块；这里该假设也失效，因为目标红物体现在只是边框，内容必须写到内部空洞
      - 原版允许图案落在红锚块周围；这里必须完全落在框内
    new_main_assumptions:
      - 每个象限恰有一个红色 4 连通方框组件
      - TL 方框内部是 2x2 非背景、非红的彩色微图案
      - 非 TL 方框内部全是背景，空洞为正方形，边长直接编码目标缩放档位
    generator_guarantees:
      - TL 永远生成 4x4 红框 + 2x2 彩色内部
      - 非 TL 永远生成 1 像素厚空心红框，且空洞边长在 1..4
      - 位置全部限制在本象限内，不越界、不碰分隔线

  placement_legality:
    - 源红框外边长固定 4，目标红框外边长为空洞边长 + 2，因此最大只到 6，始终能稳妥放入 13x13 象限
    - 每个象限只放一个红对象，不存在跨象限重叠
    - 输出填色仅写入非 TL 的空洞内部，不覆盖红框和分隔线

  visual_observability:
    - 输入里 TL 的彩色像素全部位于红框内部，源内嵌特征一眼可见
    - 输入里三个目标框内部完全空白，输出里它们被填满，目标槽位特征也一眼可见
    - 因为 2x2 源被强制非对称，TR/BL/BR 的镜像差异在 1x1、2x2、3x3、4x4 各档都能观察到

  neatness:
    - 一个象限一个框，没有外部附着图案与框叠压
    - 红框只负责标记边界，彩色内容只出现在框内，层次非常清楚

  inference_of_change:
    - 不使用颜色编号做任何隐藏逻辑；颜色只按原样复制
    - 颜色数控制在 3-4，既能保证非对称，又不会让样例显得杂乱

  post_change_assumption_audit:
    - main 中基于红组件 bbox 分象限，要求每象限只有一个红组件；generator 已保证
    - main 默认 TL 内部恰是 2x2；generator 固定源框尺寸，避免求解阶段二义性
    - main 对非 TL 直接把缩放结果写满空洞；generator 保证输入空洞初始为空背景，因此不会出现覆盖冲突
"""


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
FRAME_THICKNESS = 1
SOURCE_INNER_SIZE = 2
SOURCE_FRAME_SIZE = SOURCE_INNER_SIZE + 2
TARGET_SIZE_CHOICES = (1, 2, 3, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
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


def _reflect(img: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = img
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _has_red_square_border(patch: np.ndarray) -> bool:
    h, w = patch.shape
    if h != w or h < 3:
        return False
    return (
        np.all(patch[0, :] == RED)
        and np.all(patch[-1, :] == RED)
        and np.all(patch[:, 0] == RED)
        and np.all(patch[:, -1] == RED)
    )


def _is_hollow_red_square(patch: np.ndarray, bg: int) -> bool:
    return _has_red_square_border(patch) and bool(np.all(patch[1:-1, 1:-1] == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the miniature hidden inside the top-left red frame into the hollow red frames in the other quadrants.

    Input structure:
    - A single-color full row and full column split the 27x27 board into four 13x13 quadrants.
    - Each quadrant contains exactly one red square frame.
    - The top-left frame encloses a 2x2 colored miniature, while the other three frames enclose empty background cavities.

    Rule:
    - Extract the 2x2 miniature from inside the top-left frame.
    - For each non-top-left quadrant, reflect that miniature left/right for right quadrants and up/down for bottom quadrants.
    - Resize the reflected miniature to match the target cavity size, then fill the target frame's cavity with it.

    Key insight: Both the source and the destinations are now internal to red frames: the useful pattern is hidden inside the top-left anchor, and the other anchors contribute only empty internal slots whose sizes determine the copy scale.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = int(get_bg(g))
    mid_r, mid_c = _find_dividers(g, bg)

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red component"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert set(anchors) == set(QUADS), "need one red component per quadrant"

    sr0, sc0, sr1, sc1 = anchors[(0, 0)]
    source_frame = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert _has_red_square_border(source_frame), "top-left red object must be a square frame"
    source_inner = source_frame[1:-1, 1:-1].copy()
    assert source_inner.shape == (SOURCE_INNER_SIZE, SOURCE_INNER_SIZE), "source miniature must be 2x2"
    assert np.all(source_inner != bg), "source miniature must be visible"
    assert np.all(source_inner != RED), "source miniature must be non-red"

    for (qb, qr), (r0, c0, r1, c1) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        target_frame = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_hollow_red_square(target_frame, bg), "non-TL red object must be a hollow square frame"
        cavity_size = target_frame.shape[0] - 2 * FRAME_THICKNESS
        assert cavity_size == target_frame.shape[1] - 2 * FRAME_THICKNESS, "target cavity must be square"
        assert cavity_size >= 1, "target cavity must be non-empty"

        motif = _reflect(source_inner, qb, qr)
        motif_s = _resize_nn(motif, cavity_size, cavity_size)
        paste(out, motif_s, r0 + FRAME_THICKNESS, c0 + FRAME_THICKNESS)

    return out


def _random_inner_pattern() -> np.ndarray:
    while True:
        n_colors = random.randint(3, 4)
        palette = random.sample(COLOR_POOL, n_colors)
        cells = palette + random.choices(palette, k=4 - n_colors)
        random.shuffle(cells)
        inner = np.array(cells, dtype=int).reshape(SOURCE_INNER_SIZE, SOURCE_INNER_SIZE)
        if np.array_equal(inner, np.fliplr(inner)):
            continue
        if np.array_equal(inner, np.flipud(inner)):
            continue
        return inner


def _build_source_frame(inner: np.ndarray) -> np.ndarray:
    patch = np.full((SOURCE_FRAME_SIZE, SOURCE_FRAME_SIZE), RED, dtype=int)
    patch[1:-1, 1:-1] = inner
    return patch


def _scaled_inner(inner: np.ndarray, qb: int, qr: int, cavity_size: int) -> np.ndarray:
    return _resize_nn(_reflect(inner, qb, qr), cavity_size, cavity_size)


def _build_target_frame(cavity_size: int) -> np.ndarray:
    side = cavity_size + 2 * FRAME_THICKNESS
    frame = np.full((side, side), RED, dtype=int)
    frame[FRAME_THICKNESS:-FRAME_THICKNESS, FRAME_THICKNESS:-FRAME_THICKNESS] = BG
    return frame


def _choose_position(key: tuple[int, int], patch_h: int, patch_w: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    max_r0 = q_r1 - patch_h + 1
    max_c0 = q_c1 - patch_w + 1
    assert q_r0 <= max_r0 and q_c0 <= max_c0
    return random.randint(q_r0, max_r0), random.randint(q_c0, max_c0)


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    keys = [(0, 1), (1, 0), (1, 1)]
    while True:
        picks = [random.choice(TARGET_SIZE_CHOICES) for _ in keys]
        if len(set(picks)) < 2:
            continue
        if all(size == SOURCE_INNER_SIZE for size in picks):
            continue
        return dict(zip(keys, picks, strict=False))


def _gen_attempt() -> np.ndarray | None:
    inner = _random_inner_pattern()
    source_frame = _build_source_frame(inner)
    target_sizes = _sample_target_sizes()

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    tl_r0, tl_c0 = _choose_position((0, 0), SOURCE_FRAME_SIZE, SOURCE_FRAME_SIZE)
    paste(out, source_frame, tl_r0, tl_c0)

    frame_boxes: dict[tuple[int, int], tuple[int, int, int]] = {}
    for key, cavity_size in target_sizes.items():
        frame = _build_target_frame(cavity_size)
        fr0, fc0 = _choose_position(key, frame.shape[0], frame.shape[1])
        paste(out, frame, fr0, fc0)
        paste(out, _scaled_inner(inner, key[0], key[1], cavity_size), fr0 + FRAME_THICKNESS, fc0 + FRAME_THICKNESS)
        frame_boxes[key] = (fr0, fc0, frame.shape[0])

    inp = out.copy()
    for fr0, fc0, side in frame_boxes.values():
        inp[
            fr0 + FRAME_THICKNESS : fr0 + side - FRAME_THICKNESS,
            fc0 + FRAME_THICKNESS : fc0 + side - FRAME_THICKNESS,
        ] = BG

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
