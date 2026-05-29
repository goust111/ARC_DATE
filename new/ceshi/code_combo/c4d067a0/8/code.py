# Combo variant of c4d067a0 (index 8)
# combo_spec: D3/3+D7/3
# Name: 重力堆叠+多级缩放

import random
import sys
from functools import reduce
from math import gcd

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题: legend 像素 (rr, cc, color) 映射到主区域精确坐标
#        top_left = (base_r + rr * step, base_c + cc * step)
#      且所有颜色共用同一个 block_size。
#    - D3/3 改动后: 纵向精确坐标失效，只保留列映射
#        block_col = base_c + cc * step
#      每个图例像素对应的方块从该列顶部落下，按 rr 从小到大投放，先投的停在更低处。
#    - D7/3 改动后: 不同颜色对应不同 square size，例如红=2、蓝=3、黄=4。
#    - 融合后的小例子:
#        row0: (0,0)=红 size2, (0,1)=蓝 size3, (0,2)=黄 size4
#        row1: (1,0)=蓝 size3, (1,1)=红 size2
#        row2: (2,0)=黄 size4
#      原题会把这些像素放到规则的 rr*step 行上，且都画成同样大小。
#      融合后则是:
#        col0: 红2x2 先落底, 蓝3x3 再叠上去, 黄4x4 最后叠到最上
#        col1: 蓝3x3 先落底, 红2x2 叠上
#        col2: 黄4x4 单独落底
#      差异同时出现在两个环节:
#        1. 纵向位置由重力堆叠决定，不再是 rr*step。
#        2. 每块边长由颜色决定，不再统一。
# b) 复杂度校准
#    - 为了让“多级缩放”稳定可见，这里固定每题 3 种颜色，并把它们双射到 {2,3,4} 三种尺寸。
#    - 为了让“重力堆叠”稳定可见，legend 形状固定为 3 列顶行，且至少一列有 2 层以上。
#    - 输入只保留顶行 3 个底层锚块。这样信息足够但不冗余:
#        可见块横向相邻 -> 暴露 step/base_c
#        颜色各不相同 -> 暴露每色尺寸
#        其他块缺失 -> 必须真正执行重力补全
# c) 隐式假设审查
#    - 原 solver 依赖“所有块同尺寸”，这里失效，必须改成 size_by_color。
#    - 原 solver 依赖“块的行坐标 = base_r + rr*step”，这里失效，不能再投票 base_r。
#    - D3 单变体里的“两个锚块就够”在融合后不够，因为还要给每种颜色至少一个尺寸证据；
#      因此生成器改成保留 3 个可见底层块，且它们正好覆盖全部颜色。
#    - 若同一列上下相邻两个 legend 像素同色，则输出里会粘成一个更高的大色块，
#      使人和 solver 都难以看出层次；生成器因此禁止列内相邻层同色。
# d) 放置合法性
#    - 尺寸集合固定为 {2,3,4}，全局 step 固定为 max_size + 1 = 5，保证不同列横向至少隔 1 格。
#    - 主区域从 base_c >= 8 开始，始终与左上 legend 分区。
#    - 最高堆叠高度最多 4 层，且每层最大 4 格，总高度 <= 16；在 23-27 的画布上始终可容纳。
#    - 输入仅保留 row0 的底层块，删除其上方块后不会产生“悬空现有块”。
# e) 视觉可观察性
#    - 输入中 3 个底层锚块已经直接展示“颜色 -> 尺寸”。
#    - 输出中同一列会出现不同宽度的方块贴着堆叠，肉眼能看出“按列下落 + 不同颜色大小不同”。
#    - 因为顶行始终有 3 个相邻锚块，解题者能看出列映射仍然存在，只是纵向规则变了。
# f) 可视化整洁度
#    - legend 保持在左上孤立区域。
#    - 主区域只有 3 列活动列，列间留 1 格背景，不会横向混在一起。
#    - 尺寸变化只发生在方块边长，不额外引入标记、描边或多层编码。
# g) 变化的可推断性
#    - 不做颜色重映射；颜色本身只决定方块尺寸。
#    - 每种颜色都在输入的可见锚块里出现一次，并在 legend 里重复出现至少两次，
#      因而“这个颜色对应这个尺寸”的规律有直接证据，也有后续应用对象。
# h) main() 的新假设与 generator 保证
#    - 假设输入的可见主区域块恰好构成一段唯一的横向 legend 色序列。generator 通过:
#        1. 只让顶行宽度为 3；
#        2. 让可见块恰好对应这 3 个顶行位置；
#      来保证唯一性。
#    - 假设每种 legend 颜色都有可见尺寸样本。generator 令顶行 3 块颜色两两不同且覆盖全部颜色。
#    - 假设同列相邻层不同色。generator 回溯着色时显式保证。


def _drop_block(grid: np.ndarray, bg: int, block_size: int, c0: int, color: int) -> int:
    """Drop one solid square in a fixed main-area column and paint it in-place."""
    h, w = grid.shape
    s = int(block_size)
    c0 = int(c0)
    assert 0 <= c0 <= w - s

    def clear_at(r0: int) -> bool:
        return bool(np.all(grid[r0:r0 + s, c0:c0 + s] == bg))

    r0 = 0
    assert clear_at(r0), "drop column must be open at the top"
    while r0 + 1 <= h - s and clear_at(r0 + 1):
        r0 += 1
    grid[r0:r0 + s, c0:c0 + s] = int(color)
    return int(r0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct gravity-stacked, color-scaled block columns from a top-left legend.

    Input structure:
    - The background is the most common color.
    - The top-left corner contains singleton legend pixels on a step-2 lattice, forming a small staircase pattern.
    - The main area shows only a few bottom anchor blocks: they already rest on the floor, share one horizontal placement lattice, and different colors may have different square sizes.

    Rule:
    - Read the visible blocks to infer a color-to-size mapping and the horizontal column step in the main area.
    - Match the left-to-right visible block sequence to the unique horizontal color run in the legend; this fixes the projected main-area columns.
    - Keep the visible floor blocks in place, then for every remaining legend pixel drop a square of that color's own size into the corresponding column, processing legend rows from top to bottom so later rows stack above earlier ones.

    Key insight: the legend still controls which columns and colors exist, but vertical placement is now created by simulated falling, while block size is selected per color instead of globally.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []  # (r0, c0, size, color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0:r1 + 1, c0:c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))

    assert singles and blocks, "grid must contain legend pixels and visible anchor blocks"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    legend_color: dict[tuple[int, int], int] = {}
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
        legend_color[(int(rr), int(cc))] = int(v)

    size_by_color: dict[int, int] = {}
    visible = []
    for r0, c0, size, col in blocks:
        if col in size_by_color:
            assert size_by_color[col] == size, "each color must map to one square size"
        size_by_color[col] = int(size)
        visible.append((int(c0), int(r0), int(size), int(col)))

    legend_colors = {col for _, _, col in legend_pts}
    assert legend_colors.issubset(size_by_color), "every legend color needs a visible size sample"

    visible.sort()
    visible_colors = [col for c0, r0, size, col in visible]
    lefts = [c0 for c0, _, _, _ in visible]
    step_candidates = [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1) if lefts[i + 1] > lefts[i]]
    assert step_candidates, "visible blocks must expose the column step"
    step = reduce(gcd, step_candidates)
    assert step > max(size_by_color.values()), "projected columns must stay horizontally separated"

    anchor_matches = []
    run_len = len(visible_colors)
    for rr in sorted({rr for rr, _, _ in legend_pts}):
        row_cells = sorted([cc for rrp, cc, _ in legend_pts if rrp == rr])
        if not row_cells:
            continue
        lo = min(row_cells)
        hi = max(row_cells)
        for start_cc in range(lo, hi - run_len + 2):
            seq = [legend_color.get((rr, start_cc + k)) for k in range(run_len)]
            if any(v is None for v in seq):
                continue
            if list(seq) == visible_colors:
                anchor_matches.append((int(rr), int(start_cc)))
    assert len(anchor_matches) == 1, "visible block sequence must match a unique legend run"

    anchor_rr, anchor_cc = anchor_matches[0]
    base_c = int(visible[0][0] - anchor_cc * step)
    for i, (c0, _r0, _size, _col) in enumerate(visible):
        assert c0 == base_c + (anchor_cc + i) * step, "visible blocks must follow one coarse column lattice"

    keep = {(anchor_rr, anchor_cc + i) for i in range(run_len)}
    for cc in sorted({cc for _, cc, _ in legend_pts}):
        col_items = sorted(
            [(rr, col) for rr, ccp, col in legend_pts if ccp == cc and (rr, ccp) not in keep],
            key=lambda x: x[0],
        )
        for rr, col in col_items:
            size = size_by_color[col]
            c0 = base_c + cc * step
            assert 0 <= c0 <= w - size, "projected block column must fit in the canvas"
            _drop_block(out, bg, size, c0, col)

    return out


# --------------- generator ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
STAIRCASE_SHAPES = (
    (3, 2, 1),
    (3, 2, 2),
    (3, 1, 1, 1),
    (3, 2, 1, 1),
    (3, 2, 2, 1),
    (3, 2, 2, 2),
)
PALETTE_SIZE = 3
MIN_COUNT_PER_COLOR = 2
MAX_COUNT_PER_COLOR = 4

# === CFG: block area ===
SCALE_OPTIONS = (2, 3, 4)
BLOCK_GAP = 1
BASE_C_MIN = 8

# === CFG: generation ===
MAX_RETRIES = 500


def _pick_staircase():
    widths = random.choice(STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    return widths, pts


def _random_color_counts(total: int, k: int) -> list[int] | None:
    counts = [MIN_COUNT_PER_COLOR] * k
    remaining = total - MIN_COUNT_PER_COLOR * k
    if remaining < 0:
        return None
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        if not open_idxs:
            return None
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(widths, pts, palette):
    """Assign colors so row0 exposes all colors and no vertical neighbors share a color."""
    total = len(pts)
    counts_list = _random_color_counts(total, len(palette))
    if counts_list is None:
        return None

    row0 = [(0, c) for c in range(widths[0])]
    top_order = random.sample(palette, len(palette))
    counts = dict(zip(palette, counts_list))
    colors = {pos: col for pos, col in zip(row0, top_order)}
    for col in top_order:
        counts[col] -= 1
        if counts[col] < 0:
            return None

    cells = [p for p in pts if p not in colors]
    pos_set = set(pts)

    def allowed(pos, col):
        r, c = pos
        up = (r - 1, c)
        if up in pos_set and up in colors and colors[up] == col:
            return False
        down = (r + 1, c)
        if down in pos_set and down in colors and colors[down] == col:
            return False
        return True

    def search():
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _render_output(size: int, bg: int, step: int, base_c: int, legend_colors, size_by_color):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    placed = {}
    for rr, cc in sorted(legend_colors.keys()):
        col = legend_colors[(rr, cc)]
        c0 = base_c + cc * step
        r0 = _drop_block(out, bg, size_by_color[col], c0, col)
        placed[(rr, cc)] = (int(r0), int(c0), int(size_by_color[col]), int(col))
    return out, placed


def _has_fusion_signal(widths, legend_colors, size_by_color, placed, step: int) -> bool:
    """Reject degenerate samples where gravity or multi-scale evidence is too weak."""
    cols = sorted({cc for _, cc in legend_colors})
    gravity_seen = False
    multiscale_seen = False
    for cc in cols:
        items = sorted(
            [placed[(rr, cc)] for rr in range(len(widths)) if (rr, cc) in placed],
            key=lambda x: x[0],
            reverse=True,
        )
        sizes = [size for _r0, _c0, size, _col in items]
        if len(items) >= 2:
            diffs = [items[i][0] - items[i + 1][0] for i in range(len(items) - 1)]
            if any(diff != step for diff in diffs):
                gravity_seen = True
            if len(set(sizes)) >= 2:
                multiscale_seen = True
    return gravity_seen and multiscale_seen


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    widths, pts = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], PALETTE_SIZE)
    legend_colors = _assign_legend_colors(widths, pts, palette)
    if legend_colors is None:
        return None

    size_perm = random.sample(SCALE_OPTIONS, len(palette))
    size_by_color = {col: block_size for col, block_size in zip(palette, size_perm)}
    step = max(size_by_color.values()) + BLOCK_GAP

    max_c = max(c for _, c in pts)
    max_size = max(size_by_color.values())
    c_hi = size - max_size - max_c * step
    if c_hi < BASE_C_MIN:
        return None
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, placed = _render_output(size, bg, step, base_c, legend_colors, size_by_color)
    if not _has_fusion_signal(widths, legend_colors, size_by_color, placed, step):
        return None

    inp = out.copy()
    keep = {(0, 0), (0, 1), (0, 2)}
    for pos, (r0, c0, block_size, _col) in placed.items():
        if pos in keep:
            continue
        inp[r0:r0 + block_size, c0:c0 + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
