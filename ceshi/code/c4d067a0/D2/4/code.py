# Variant of c4d067a0
# Dimension: D2 — 源对象：图例的结构与位置 (src)
# Change: 左上角稀疏像素模式（非背景色像素以偶数间隔排列在奇数行/列上），每个像素代表一个待放大的色块...
#   → 对角线标记对作为图例
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题左上角图例直接把每个目标块编码成一个单像素；这里改成“每种颜色恰好两个单像素端点”，两点位于同一小矩形的对角。
#    - 图例本身不再显示矩形内部，只显示角点；主区域输出则仍然是规则块网格，但要填的是“该小矩形覆盖的所有格子”。
#    - 例子：若图例里某色端点在格点坐标 (0,0) 与 (1,1)，原题只会对应 2 个块；新题要把这个 2x2 小矩形的 4 个格子都投影成块。
#    - 因而流程变化发生在“从 legend 解析待投影的格点集合”这一步：原逻辑是单点列表；新逻辑是先按颜色配对，再把每对对角扩展成一个轴对齐小矩形。
# b) 复杂度校准
#    - 若仍只保留原题的 2 个输入块，则一个 2x2/2x3 图例矩形内部常有多个相同的相邻块对，偏移会多解。
#    - 为保持 D2 变化可见且可解，输入最小幅度地改为 3 个已知块：
#      1) 同色相邻 2 块，用来锁定 block step；
#      2) 另一种颜色的 1 块，用来打破同色矩形内部的平移歧义。
#    - 每个样例使用 2 个图例矩形，面积总数控制在 8 或 10 个格点，既明显不同于原题，又不会太乱。
# c) 隐式假设审查（对应 main 的关键依赖）
#    - 组件提取仍假设：图例是单像素连通块，主区域是纯色正方形块。生成器保证这一点。
#    - 原题把每个 legend 单像素直接当作一个目标格点；现在这条前提失效，改为“每种 legend 颜色恰有 2 个单像素端点”。
#    - 原题可只靠颜色投票定位 base；现在同色会展开成多个格点，因此单靠 2 块会歧义。生成器保证 3 块锚点使 base 投票唯一。
#    - 原题默认 legend 点阵的 gcd 就是格距；这里端点仍放在 step=2 的稀疏格点上，并保证整体行列差的 gcd 仍为 2。
# d) 放置合法性
#    - 图例矩形在 legend 的格点坐标里必须互不重叠，否则主区域会要求同一块位置出现两种颜色。
#    - 所有投影块使用统一 block_size 与 step=block_size+gap，且 legend 格点集合无重复，因此主区域块也不会重叠。
#    - `base_c >= 8` 继续把块区放到图例右侧；实际放置前按网格大小检查是否全体 fit。
# e) 视觉可观察性
#    - 人看到的是：左上角不再是已填满的小地图，而是每种颜色只有两个对角端点；输出却补出整块矩形区域。
#    - 2 个矩形足以在 2-3 个样例中稳定展示“角点定义矩形”的规律，且颜色不参与任何不可见排序。
# f) 可视化整洁度
#    - 图例只占左上 4x4 的稀疏格点窗口，主块区整体右移，避免与图例混在一起。
#    - 图例矩形数固定为 2，颜色也固定为 2 种，视觉层次简单。
# g) 颜色变化可推断性
#    - 不存在颜色映射或重编码：图例端点颜色就是对应主块矩形的颜色。
#    - 每种颜色只对应一个矩形，避免“同色多矩形”造成的人类配对歧义。
# h) 变体判据
#    - 新行为的核心量化指标是“每种 legend 颜色只出现 2 个端点，但输出块格点数为该矩形面积(4 或 6)，显著大于 2”。
#    - 若生成样本里所有矩形面积都退化成 1，则会退回原题；生成器禁止这种情况。
# i) main 的新假设与 generator 保障
#    - 假设每种 legend 颜色在左上窗口里恰好出现 2 次：generator 强制如此。
#    - 假设这两个端点确为对角而非同行/同列：generator 只放 2x2 / 2x3 / 3x2 的矩形。
#    - 假设 3 个保留块能给出唯一 base：generator 先在格点层面检查 base 投票唯一，再调用 main 终验。

import sys
from functools import reduce
from math import gcd
import random

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, components_mask, get_bg


def _gcd_nonzero(vals: list[int]) -> int:
    vals = [abs(int(v)) for v in vals if int(v) != 0]
    assert vals, "need at least one non-zero difference"
    return reduce(gcd, vals)


def _extract_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in components_mask(g != bg, connectivity=4):
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _legend_step(markers: list[tuple[int, int, int]]) -> int:
    rows = sorted({r for r, _, _ in markers})
    cols = sorted({c for _, c, _ in markers})
    diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    diffs += [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    return _gcd_nonzero(diffs)


def _legend_rectangles(markers: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in markers:
        by_color.setdefault(int(col), []).append((int(r), int(c)))
    assert by_color, "legend must contain marker colors"
    assert all(len(pts) == 2 for pts in by_color.values()), "each legend color must have exactly two markers"

    stepL = _legend_step(markers)
    rmin = min(r for r, _, _ in markers)
    cmin = min(c for _, c, _ in markers)

    rects: list[tuple[int, int, int, int, int]] = []
    for col, pts in by_color.items():
        (r1, c1), (r2, c2) = pts
        rr1 = (r1 - rmin) // stepL
        cc1 = (c1 - cmin) // stepL
        rr2 = (r2 - rmin) // stepL
        cc2 = (c2 - cmin) // stepL
        assert rr1 != rr2 and cc1 != cc2, "legend markers must be diagonal"
        r0 = min(rr1, rr2)
        c0 = min(cc1, cc2)
        h = abs(rr1 - rr2) + 1
        w = abs(cc1 - cc2) + 1
        rects.append((r0, c0, h, w, int(col)))
    return rects


def _expand_rectangles(rects: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r0, c0, h, w, col in rects:
        for rr in range(r0, r0 + h):
            for cc in range(c0, c0 + w):
                cells.append((int(rr), int(cc), int(col)))
    return cells


def _block_step(blocks: list[tuple[int, int, int, int]]) -> int:
    diffs: list[int] = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            diffs.append(abs(blocks[i][0] - blocks[j][0]))
            diffs.append(abs(blocks[i][1] - blocks[j][1]))
    return _gcd_nonzero(diffs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block grid from diagonal-pair legend markers in the top-left corner.

    Input structure:
    - bg is the dominant color.
    - Top-left legend: each non-bg color appears as exactly two singleton pixels on a sparse step-2 lattice.
    - Each same-colored singleton pair sits at opposite corners of a small rectangle in legend coordinates.
    - Main area: a few uniformly colored square blocks already exist on a larger regular lattice.

    Rule:
    - Group the legend singletons by color and pair each color's two diagonal endpoints.
    - Expand each paired endpoint box into all lattice cells inside that rectangle.
    - Use the existing large blocks to infer the block scale and translation from legend-lattice cells to block positions.
    - Paint the missing square blocks for every cell covered by every legend rectangle.

    Key insight: The legend no longer lists target cells explicitly; each color contributes only two corner markers, and the hidden rectangle between them must be reconstructed before projecting it into the main block lattice.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend markers and main blocks"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, col) for (r, c, col) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton markers"

    rects = _legend_rectangles(legend)
    legend_cells = _expand_rectangles(rects)

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant size"

    step = _block_step(blocks)
    assert step > 0, "block step must be positive"

    visible_by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _, col in blocks:
        visible_by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    legend_by_color: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, col in legend_cells:
        legend_by_color.setdefault(int(col), []).append((int(rr), int(cc)))

    bases: dict[tuple[int, int], int] = {}
    for col, cells in legend_by_color.items():
        if col not in visible_by_color:
            continue
        for rr, cc in cells:
            for br0, bc0 in visible_by_color[col]:
                base = (br0 - rr * step, bc0 - cc * step)
                bases[base] = bases.get(base, 0) + 1
    assert bases, "visible blocks must determine a base offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-grid offset must be unique"
    base_r, base_c = best_bases[0]

    for rr, cc, col in legend_cells:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "projected block must fit in the canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (D2) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_LATTICE_H = 4
LEGEND_LATTICE_W = 4
RECT_SHAPES = ((2, 2), (2, 3), (3, 2))
NUM_RECTS = 2

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 500


def _rect_cells(rect: tuple[int, int, int, int, int, str]) -> list[tuple[int, int]]:
    r0, c0, h, w, _col, _diag = rect
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _rect_markers(rect: tuple[int, int, int, int, int, str]) -> list[tuple[int, int, int]]:
    r0, c0, h, w, col, diag = rect
    if diag == "TLBR":
        pts = [(r0, c0), (r0 + h - 1, c0 + w - 1)]
    else:
        pts = [(r0, c0 + w - 1), (r0 + h - 1, c0)]
    return [(int(r), int(c), int(col)) for r, c in pts]


def _all_legend_cells(rects: list[tuple[int, int, int, int, int, str]]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for rect in rects:
        r0, c0, h, w, col, _diag = rect
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                cells.append((int(r), int(c), int(col)))
    return cells


def _base_votes_in_lattice(rects: list[tuple[int, int, int, int, int, str]],
                           anchors: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, col in _all_legend_cells(rects):
        by_color.setdefault(int(col), []).append((int(rr), int(cc)))

    votes: dict[tuple[int, int], int] = {}
    for vr, vc, col in anchors:
        for rr, cc in by_color[int(col)]:
            base = (int(vr) - int(rr), int(vc) - int(cc))
            votes[base] = votes.get(base, 0) + 1
    return votes


def _anchor_triplet(rects: list[tuple[int, int, int, int, int, str]]) -> list[tuple[int, int, int]] | None:
    by_rect_cells = [(_rect_cells(rect), rect[4]) for rect in rects]
    order = list(range(len(rects)))
    random.shuffle(order)
    for idx in order:
        cells_a, col_a = by_rect_cells[idx]
        adj_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        cell_set = set(cells_a)
        for rr, cc in cells_a:
            for dr, dc in ((0, 1), (1, 0)):
                nxt = (rr + dr, cc + dc)
                if nxt in cell_set:
                    adj_pairs.append(((rr, cc), nxt))
        random.shuffle(adj_pairs)
        other_idxs = [j for j in range(len(rects)) if j != idx]
        random.shuffle(other_idxs)
        for a, b in adj_pairs:
            for j in other_idxs:
                cells_b, col_b = by_rect_cells[j]
                probe_cells = cells_b[:]
                random.shuffle(probe_cells)
                for rr, cc in probe_cells:
                    anchors = [
                        (int(a[0]), int(a[1]), int(col_a)),
                        (int(b[0]), int(b[1]), int(col_a)),
                        (int(rr), int(cc), int(col_b)),
                    ]
                    votes = _base_votes_in_lattice(rects, anchors)
                    max_votes = max(votes.values())
                    best = [base for base, cnt in votes.items() if cnt == max_votes]
                    if max_votes == len(anchors) and best == [(0, 0)]:
                        return anchors
    return None


def _pick_rectangles(colors: list[int]) -> list[tuple[int, int, int, int, int, str]] | None:
    shapes = list(RECT_SHAPES)
    random.shuffle(shapes)
    for _ in range(80):
        chosen_shapes = [random.choice(shapes) for _ in range(NUM_RECTS)]
        if sum(h * w for h, w in chosen_shapes) not in (8, 10):
            continue
        rects: list[tuple[int, int, int, int, int, str]] = []
        occupied: set[tuple[int, int]] = set()
        ok = True
        for (h, w), col in zip(chosen_shapes, colors):
            placements = []
            for r0 in range(LEGEND_LATTICE_H - h + 1):
                for c0 in range(LEGEND_LATTICE_W - w + 1):
                    cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
                    if cells & occupied:
                        continue
                    for diag in ("TLBR", "TRBL"):
                        placements.append((r0, c0, h, w, col, diag))
            if not placements:
                ok = False
                break
            rect = random.choice(placements)
            rects.append(rect)
            occupied |= set(_rect_cells(rect))
        if ok:
            return rects
    return None


def _render_output(size: int,
                   bg: int,
                   rects: list[tuple[int, int, int, int, int, str]],
                   block_size: int,
                   step: int,
                   base_r: int,
                   base_c: int) -> np.ndarray:
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    for r, c, col in [_pt for rect in rects for _pt in _rect_markers(rect)]:
        out[lr0 + r * LEGEND_STEP, lc0 + c * LEGEND_STEP] = col
    for rr, cc, col in _all_legend_cells(rects):
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size = random.choice(tuple(BLOCK_SIZE_RANGE))
    gap = random.choice(tuple(GAP_RANGE))
    step = block_size + gap
    if not (STEP_MIN <= step <= STEP_MAX):
        return None

    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], NUM_RECTS)
    rects = _pick_rectangles(palette)
    if rects is None:
        return None

    anchors = _anchor_triplet(rects)
    if anchors is None:
        return None

    all_cells = _all_legend_cells(rects)
    max_r = max(r for r, _, _ in all_cells)
    max_c = max(c for _, c, _ in all_cells)
    r_hi = size - block_size - max_r * step
    c_hi = size - block_size - max_c * step
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out = _render_output(size, bg, rects, block_size, step, base_r, base_c)
    inp = out.copy()

    keep = {(rr, cc, col) for rr, cc, col in anchors}
    for rr, cc, col in all_cells:
        if (rr, cc, col) in keep:
            continue
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

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
    raise RuntimeError("failed to generate valid c4d067a0 D2 input")
