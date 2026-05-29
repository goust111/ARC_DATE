# Combo variant of da515329 (index 32)
# combo_spec: D6/1+D8/2+D14/2
# Name: 同心矩形框 + 保留十字 + 交替涂色

import math
import random

import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the combo variant
# -------------------------------------
# 1) 具体化与行为追踪
#    - 输入仍是单个青色(8)中心空洞十字，四臂等长，背景为黑色(0)。
#    - 原题会从四个端点各发射一条连续顺时针螺旋，并靠 turn/step/spacing 规则把
#      全图织成一条单像素宽迷宫路径。
#    - 本融合不再追踪连续路径，而是把“四端点围出的最小正方框”视为第 0 圈，
#      再每次向外扩 2 格得到第 1、2、3... 圈离散矩形框。扩 2 而不是扩 1，
#      保留了原题 1 像素间距的视觉节奏。
#    - 同时保留原十字本体：输出先复制输入，再补画同心框。若某个框经过十字端点，
#      那些源像素仍保持原值 8，不被改写。
#    - 颜色上不再是单色螺旋。第 0 圈用青色(8)，第 1 圈用红色(2)，之后按圈号
#      8/2/8/2... 交替。这样十字自然并入最内层青框，外层再出现红框，层次清楚。
#    - 例：arm=2 时，输入端点围出 5x5 框。原题会在 5x5 框外继续长成一条 maze；
#      本题则直接画 5x5 青框、9x9 红框、13x13 青框……，中心十字仍完整可见。
#
# 2) 复杂度校准
#    - 只保留一个对象类型（中心空洞十字），输出只增加“规则化的矩形框 + 固定双色”。
#      自由度主要来自 arm 长度和十字在近中心位置的偏移，足以产生多样样例但不会
#      引入难以归纳的对象匹配问题。
#    - 若按“沿框周长逐像素交替”上色，最内圈会在四边形成碎花纹，和保留十字叠加后
#      容易显得杂乱。这里采用“按整圈交替”来落实 D14/2 的双色思想，让人更容易
#      在少量样例里看出“外扩一圈就换一种颜色”的规律。
#    - 生成器强制至少 3 圈，这样每个样例都同时展示：离散同心框、源十字保留、
#      两色都真实出现，而不是退化成单圈或单色。
#
# 3) 隐式假设审查
#    - 原 main() 的 agent 状态、右转优先、1/2 步长、边界补丁、奇偶后处理，全都
#      假设输出是一条连续自避路径。改成离散框以后这些前提全部失效，必须整体删除。
#    - 新 solver 依赖的新前提更简单：
#      * 唯一显著前景仍是一个等臂十字；
#      * 十字中心可由“青色数量最多的行/列”恢复；
#      * 框以 arm_len 为半径向外每次扩 2 格即可。
#    - `main()` 的每个循环只遍历这些矩形框，不再依赖任何路径连通性或边界修补。
#    - keep_source 引入的新假设是：输入里没有别的青色对象，否则“保护所有 8 像素”
#      可能误保留杂质。生成器继续只造单个十字来保证这一点。
#
# 4) 放置合法性
#    - 生成器只在正方形画布上放置一个十字，且中心选在近中央的合法位置。
#    - 合法位置要求至少能容纳 3 圈完整矩形框；所有框都由 bbox 公式直接算出，
#      不会越界，也不会互相重叠成厚边，因为每圈 pad 都加 2。
#    - 源十字与第 0 圈只在四个端点接触，不会和更外层框重叠。
#
# 5) 视觉可观察性 / 整洁度 / 颜色可推断性
#    - 人眼可以直接从样例看出：输入里的十字还在；输出是多圈正方框；颜色沿圈层
#      交替，而不是任意散布。
#    - 输出只用背景黑、源青、辅助红三种颜色，颜色角色稳定：青色总和“源十字 +
#      偶数圈”绑定，红色总和奇数圈绑定，不依赖颜色编号大小等不可视因素。
#    - 1 像素框 + 1 像素黑间隔使不同圈层不会糊在一起；十字只占中心十字形通道，
#      不会把整幅图变成难分辨的覆盖体。
#
# 6) 新增假设与 generator 保证
#    - `_find_cross` 假设四臂等长且轴对齐；生成器直接按该模板构造。
#    - `_frame_rectangles` 假设画布足够大以容纳至少 1 圈；生成器进一步保证 >=3 圈。
#    - `_fusion_metrics` 假设目标输出应含两种前景色、源像素逐点保留、且每一圈
#      除源端点外是单色；生成器用这些指标筛掉退化样本。


CROSS_COLOR = 8
ALT_COLOR = 2


def _find_cross(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
    """Return center-hole row/col, directional arm cells, and common arm length."""
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _frame_rectangles(
    h: int, w: int, cr: int, cc: int, arm_len: int
) -> list[tuple[int, int, int, int]]:
    """List all fully in-bounds concentric frames starting from the tip bbox."""
    rects = []
    pad = 0
    while True:
        top = cr - arm_len - pad
        bottom = cr + arm_len + pad
        left = cc - arm_len - pad
        right = cc + arm_len + pad
        if top < 0 or left < 0 or bottom >= h or right >= w:
            break
        rects.append((top, left, bottom, right))
        pad += 2
    return rects


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    """Return a boolean 1-pixel perimeter mask for one rectangle."""
    top, left, bottom, right = rect
    mask = np.zeros(shape, dtype=bool)
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True
    return mask


def _ring_color(index: int) -> int:
    """Even rings keep the source teal; odd rings use the alternate color."""
    return CROSS_COLOR if index % 2 == 0 else ALT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the original spiral with spaced concentric rectangular frames around a preserved cross.

    Input structure:
    - bg = 0 (black). A single teal (8) plus-sign has four equal arms and an empty center hole.

    Rule:
    - Find the cross center from the densest teal row/column and infer the common arm length from the four arm tips.
    - Use the tip-bounding square as the first 1-pixel rectangular frame, then expand outward by 2 cells at a time to make further disjoint frames.
    - Keep every original teal cross pixel unchanged, and color the frames by ring: teal for the innermost ring, red for the next, then alternate teal/red for each outer ring.

    Key insight: Once the spiral is discretized into its square envelopes, the whole task reduces to reading the cross radius and painting evenly spaced rectangular shells while protecting the source cross.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    cr, cc, _arms, arm_len = _find_cross(grid)
    if arm_len == 0:
        return out

    source_mask = grid == CROSS_COLOR
    rects = _frame_rectangles(*grid.shape, cr, cc, arm_len)
    for idx, rect in enumerate(rects):
        frame_mask = _rect_mask(grid.shape, rect)
        out[frame_mask & ~source_mask] = _ring_color(idx)

    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
MAX_OUTER_RETRIES = 200
MIN_RING_COUNT = 3
MIN_ALT_COLOR_PIXELS = 12
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build a square grid with a plus-sign cross (center hole) at (cr, cc)."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _candidate_centers(size: int, arm: int) -> list[tuple[int, int]]:
    """
    Prefer near-central placements that preserve several complete rings.

    This keeps the discrete-frame structure obvious and reduces one-sided truncation.
    """
    scored = []
    mid = (size - 1) / 2.0
    for cr in range(arm, size - arm):
        for cc in range(arm, size - arm):
            ring_count = len(_frame_rectangles(size, size, cr, cc, arm))
            if ring_count < MIN_RING_COUNT:
                continue
            centrality = abs(cr - mid) + abs(cc - mid)
            scored.append((centrality, cr, cc))

    scored.sort()
    return [(cr, cc) for _score, cr, cc in scored[:8]]


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    """Quantify the core fused behaviors for generator filtering and post-run audits."""
    cr, cc, _arms, arm_len = _find_cross(inp)
    rects = _frame_rectangles(*inp.shape, cr, cc, arm_len)
    source_mask = inp == CROSS_COLOR

    alternating_rings = 0
    for idx, rect in enumerate(rects):
        ring_mask = _rect_mask(inp.shape, rect) & ~source_mask
        if not np.any(ring_mask):
            alternating_rings += 1
            continue
        vals = set(map(int, np.unique(out[ring_mask])))
        if vals == {_ring_color(idx)}:
            alternating_rings += 1

    return {
        "ring_count": int(len(rects)),
        "alternating_rings": int(alternating_rings),
        "alt_color_pixels": int(np.sum(out == ALT_COLOR)),
        "source_preserved": int(np.array_equal(out[source_mask], inp[source_mask])),
        "changed_pixels": int(np.sum(out != inp)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Pick an arm length and a legal near-central placement, then validate the fused behavior."""
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        metrics = _fusion_metrics(inp, out)
        if metrics["ring_count"] < MIN_RING_COUNT:
            continue
        if metrics["alternating_rings"] != metrics["ring_count"]:
            continue
        if metrics["source_preserved"] != 1:
            continue
        if metrics["alt_color_pixels"] < MIN_ALT_COLOR_PIXELS:
            continue
        if metrics["changed_pixels"] <= 0:
            continue
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
