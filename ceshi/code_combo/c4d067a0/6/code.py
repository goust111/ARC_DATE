# Combo variant of c4d067a0 (index 6)
# combo_spec: D2/4+D4/4
# Name: 对角线标记图例+分形自相似放大

import random
import sys
from functools import reduce
from math import gcd

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg


# Design analysis
# a) 具体化与行为追踪
#    - 原题左上角图例直接列出“哪些格点要被放大”，主区域里每个目标格点对应一个实心同色方块。
#    - 这里先做 D2：图例不再显式画出全部格点，而是每种颜色只给两个单像素端点；这两个端点位于同一小矩形的对角，解题时必须先把矩形内部格点补出来。
#    - 再做 D4：主区域里的每个目标宏方块也不再是实心块，而是把“所有矩形展开后的完整格点轮廓”缩小嵌到方块内部，并统一涂成驱动格点的颜色。
#    - 例子：若红色端点定义 2x2 矩形、蓝色端点定义另一个 2x3 矩形，则真正的 legend 不是 4 个端点，而是 4+6 个格点的并集；其中任一红色格点映射到主区后，得到的是一个红色的稀疏 mini-legend，而不是红色实心块。
# b) 复杂度校准
#    - 若只保留 2 个主区锚块，则同色矩形内部会给出多个可平移匹配，base 容易多解。
#    - 因此输入保留 3 个分形锚瓦片：同一矩形内一对相邻同色格点，用来锁定 coarse step；另一矩形再保留 1 个异色格点，用来打破同色平移歧义。
#    - 图例固定为 2 个非重叠矩形，面积总和控制在 8 或 10，既足够让“端点定义矩形”显眼，又让嵌入到每个宏方块里的轮廓稳定可见。
# c) 隐式假设审查
#    - 原 solver 假设主区对象是 4 连通实心正方形；D4 后这条失效，所以不能再用连通块直接读 block_size，必须改成“在主区搜索同色 mini-legend 轮廓”。
#    - 原 solver 把 legend 单像素直接当目标格点；D2 后这条失效，所以要先按颜色配对，再展开矩形覆盖格点。
#    - 原 D4 单变体要求 legend 轮廓能触到宏方块的左上与右下边界，便于从同色像素反推出宏方块左上角；这里生成器显式保证展开后的格点并集覆盖归一化后的 (0,0) 与右下角。
#    - 原题用已有实心块的 top-left gcd 推 step；这里改为已识别分形瓦片的 top-left gcd，且生成器保证至少有一对相邻格点作为锚，从而 gcd 真正等于 step 而不是其倍数。
# d) 放置合法性
#    - 两个图例矩形在 4x4 的 legend lattice 内必须互不重叠，否则同一 legend 格点会被两色同时占用。
#    - 宏方块使用统一 `block_size` 与 `step=block_size+1`，分形瓦片只画在各自方块内部；方块之间保留 1 格间隔，所以不会互相污染。
#    - `base_c >= 12` 把主区推到 legend 窗口右侧；同时根据最远 legend 格点检查所有宏方块都完整落在画布内。
# e) 视觉可观察性
#    - 人看到的输入会有两个明显线索：左上角只有 4 个对角端点，而主区出现的少数锚对象不是实心块，而是同一稀疏轮廓的重复。
#    - 输出中缺失对象补齐后，解题者可以同时观察到“端点补成矩形”与“每个矩形格点又展开成同一 mini-legend 轮廓”。
# f) 可视化整洁度
#    - legend 只占左上 8x8 像素内的稀疏位置，主区整体右移，避免与分形瓦片混在一起。
#    - 只使用 2 个矩形、2 种非背景色，且宏方块间固定留白，视觉层次清晰。
# g) 变化的可推断性
#    - 不做颜色重映射：端点颜色就是该矩形内部格点颜色，某个格点再驱动对应颜色的分形瓦片。
#    - 分形瓦片内部的非背景像素全部同色，颜色规则完全由外层格点决定，不依赖颜色 id 的不可视排序。
# h) 融合判据
#    - 关键指标 1：每种颜色在 legend 里只出现 2 个端点，但展开后的同色目标格点数必须大于 2。
#    - 关键指标 2：任一已保留主区对象的填充率都必须严格小于 1，说明它不是原题那种实心块。
#    - 若样本同时满足这两点，且按原题“端点即单点、主区实心块”解释会得到不同输出，就说明融合行为确实发生。
# i) main 的新假设与 generator 保障
#    - 假设每种 legend 颜色恰好 2 个标记点：generator 强制每个颜色只对应一个矩形。
#    - 假设端点确为对角而非同行同列：generator 只放 2x2 / 2x3 / 3x2 矩形。
#    - 假设展开后的格点并集含有归一化后的左上角与右下角：generator 只接受覆盖 (0,0) 和 (3,3) 的矩形组合。
#    - 假设输入中至少能识别 3 个分形锚瓦片且 base 唯一：generator 先在 legend lattice 上做投票检查，再调用 `main()` 终验。


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 35, 39
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_WINDOW = 12
LEGEND_LATTICE_H = 4
LEGEND_LATTICE_W = 4
RECT_SHAPES = ((2, 2), (2, 3), (3, 2))
NUM_RECTS = 2

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(4, 6)
BLOCK_GAP = 1
BASE_C_MIN = 12
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 700


def _gcd_nonzero(vals: list[int]) -> int:
    vals = [abs(int(v)) for v in vals if int(v) != 0]
    assert vals, "need at least one non-zero difference"
    return reduce(gcd, vals)


def _spread_positions(count: int, size: int) -> list[int]:
    if count == 1:
        return [0]
    assert size >= count, "mini-legend must fit inside macro square"
    return [(i * (size - 1)) // (count - 1) for i in range(count)]


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
    for r0, c0, h, w, col, _diag in rects:
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                cells.append((int(r), int(c), int(col)))
    cells.sort()
    return cells


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
    assert len(by_color) == NUM_RECTS, "legend must contain one color per rectangle"
    assert all(len(pts) == 2 for pts in by_color.values()), "each legend color must have exactly two markers"

    stepL = _legend_step(markers)
    rmin = min(r for r, _, _ in markers)
    cmin = min(c for _, c, _ in markers)

    rects: list[tuple[int, int, int, int, int]] = []
    for col, pts in sorted(by_color.items()):
        (r1, c1), (r2, c2) = pts
        rr1 = (r1 - rmin) // stepL
        cc1 = (c1 - cmin) // stepL
        rr2 = (r2 - rmin) // stepL
        cc2 = (c2 - cmin) // stepL
        assert rr1 != rr2 and cc1 != cc2, "legend markers must define a diagonal"
        r0 = min(rr1, rr2)
        c0 = min(cc1, cc2)
        h = abs(rr1 - rr2) + 1
        w = abs(cc1 - cc2) + 1
        rects.append((int(r0), int(c0), int(h), int(w), int(col)))
    return rects


def _expand_rectangles(rects: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r0, c0, h, w, col in rects:
        for rr in range(r0, r0 + h):
            for cc in range(c0, c0 + w):
                cells.append((int(rr), int(cc), int(col)))
    cells.sort()
    return cells


def _tile_relative_points(legend_cells: list[tuple[int, int, int]], block_size: int) -> list[tuple[int, int]]:
    rows = max(rr for rr, _, _ in legend_cells) + 1
    cols = max(cc for _, cc, _ in legend_cells) + 1
    row_pos = _spread_positions(rows, block_size)
    col_pos = _spread_positions(cols, block_size)
    rel = sorted({(row_pos[rr], col_pos[cc]) for rr, cc, _ in legend_cells})
    assert (0, 0) in rel, "embedded legend must touch macro-square top-left"
    assert max(r for r, _ in rel) == block_size - 1, "embedded legend must touch macro-square bottom"
    assert max(c for _, c in rel) == block_size - 1, "embedded legend must touch macro-square right"
    return rel


def _paint_fractal_tile(
    out: np.ndarray,
    r0: int,
    c0: int,
    block_size: int,
    rel_pts: list[tuple[int, int]],
    col: int,
    bg: int,
) -> None:
    out[r0 : r0 + block_size, c0 : c0 + block_size] = bg
    for dr, dc in rel_pts:
        out[r0 + dr, c0 + dc] = col


def _extract_legend_markers(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    win_r = min(h, LEGEND_WINDOW)
    win_c = min(w, LEGEND_WINDOW)
    markers = [
        (r, c, int(g[r, c]))
        for r in range(win_r)
        for c in range(win_c)
        if int(g[r, c]) != bg
    ]
    assert markers, "top-left legend must contain colored markers"
    return markers


def _block_step(found: list[tuple[int, int, int, int]]) -> int:
    diffs: list[int] = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            diffs.append(found[i][0] - found[j][0])
            diffs.append(found[i][1] - found[j][1])
    return _gcd_nonzero(diffs)


def _find_fractal_tiles(
    g: np.ndarray,
    bg: int,
    legend_abs: set[tuple[int, int]],
    legend_cells: list[tuple[int, int, int]],
) -> tuple[int, int, tuple[int, int], list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    by_color: dict[int, set[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val == bg or (r, c) in legend_abs:
                continue
            by_color.setdefault(val, set()).add((r, c))
    assert by_color, "grid must contain fractal macro tiles"

    legend_by_color: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, col in legend_cells:
        legend_by_color.setdefault(int(col), []).append((int(rr), int(cc)))

    best = None
    for block_size in BLOCK_SIZE_RANGE:
        rel_pts = _tile_relative_points(legend_cells, block_size)
        found: list[tuple[int, int, int, int]] = []
        for col, coords in by_color.items():
            for r0, c0 in sorted(coords):
                if r0 + block_size > h or c0 + block_size > w:
                    continue
                needed = {(r0 + dr, c0 + dc) for dr, dc in rel_pts}
                if not needed.issubset(coords):
                    continue
                ok = True
                for rr in range(r0, r0 + block_size):
                    for cc in range(c0, c0 + block_size):
                        val = int(g[rr, cc])
                        if (rr, cc) in needed:
                            if val != col:
                                ok = False
                                break
                        elif val != bg:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    found.append((int(r0), int(c0), int(block_size), int(col)))
        if len(found) < 3:
            continue

        step = _block_step(found)
        if step <= 0:
            continue

        placements_by_color: dict[int, list[tuple[int, int]]] = {}
        for r0, c0, _s, col in found:
            placements_by_color.setdefault(int(col), []).append((int(r0), int(c0)))

        bases: dict[tuple[int, int], int] = {}
        for col, cells in legend_by_color.items():
            if col not in placements_by_color:
                continue
            for rr, cc in cells:
                for br0, bc0 in placements_by_color[col]:
                    base = (int(br0) - int(rr) * step, int(bc0) - int(cc) * step)
                    bases[base] = bases.get(base, 0) + 1
        if not bases:
            continue
        max_votes = max(bases.values())
        best_bases = [base for base, votes in bases.items() if votes == max_votes]
        if max_votes != len(found) or len(best_bases) != 1:
            continue

        candidate = (len(found), max_votes, block_size, step, best_bases[0], found, rel_pts)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None, "failed to identify fractal macro tiles"
    _count, _votes, block_size, step, base, found, rel_pts = best
    return int(block_size), int(step), tuple(map(int, base)), found, rel_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a coarse block layout from diagonal-pair legend markers and self-similar macro tiles.

    Input structure:
    - The background is the most common color.
    - The top-left legend contains exactly two colors, and each color appears as two singleton markers on a step-2 lattice.
    - Each same-colored marker pair lies on opposite corners of a hidden rectangle in legend coordinates.
    - Elsewhere, a few already-placed macro tiles appear; each tile is a square whose colored pixels form a uniformly recolored miniature copy of the full expanded legend shape.

    Rule:
    - Group the legend markers by color, pair each color's two diagonal endpoints, and expand every pair into all lattice cells inside its rectangle.
    - Treat the union of those expanded cells as the true legend shape.
    - Find the existing macro tiles by matching that same shape, shrunk into a square and recolored with one single tile color.
    - Use the visible macro tiles to infer the macro-square size, the coarse lattice step, and the translation from legend cells to main-area placements.
    - For every expanded legend cell, project its lattice position into the main area and draw the corresponding self-similar tile in that cell's color.

    Key insight: The legend must be decoded twice: first the diagonal markers hide full rectangles, then that reconstructed multi-rectangle pattern reappears inside every main-area tile as a one-color fractal miniature, and those repeated miniatures reveal the global placement map.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    legend_markers = _extract_legend_markers(g, bg)
    rects = _legend_rectangles(legend_markers)
    legend_cells = _expand_rectangles(rects)
    legend_abs = {(r, c) for r, c, _ in legend_markers}

    block_size, step, (base_r, base_c), _found, rel_pts = _find_fractal_tiles(
        g, bg, legend_abs, legend_cells
    )

    for rr, cc, col in legend_cells:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "projected macro tile must fit inside the canvas"
        )
        _paint_fractal_tile(out, tr0, tc0, block_size, rel_pts, col, bg)
    return out


def _base_votes_in_lattice(
    legend_cells: list[tuple[int, int, int]],
    anchors: list[tuple[int, int, int]],
) -> dict[tuple[int, int], int]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for rr, cc, col in legend_cells:
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
        cell_set = set(cells_a)
        adj_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
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
                    votes = _base_votes_in_lattice(_all_legend_cells(rects), anchors)
                    max_votes = max(votes.values())
                    best = [base for base, cnt in votes.items() if cnt == max_votes]
                    if max_votes == len(anchors) and best == [(0, 0)]:
                        return anchors
    return None


def _pick_rectangles(colors: list[int]) -> list[tuple[int, int, int, int, int, str]] | None:
    for _ in range(160):
        chosen_shapes = [random.choice(RECT_SHAPES) for _ in range(NUM_RECTS)]
        if sum(h * w for h, w in chosen_shapes) not in (8, 10):
            continue

        placements_by_shape: list[list[tuple[int, int, int, int, int, str]]] = []
        for (h, w), col in zip(chosen_shapes, colors):
            placements: list[tuple[int, int, int, int, int, str]] = []
            for r0 in range(LEGEND_LATTICE_H - h + 1):
                for c0 in range(LEGEND_LATTICE_W - w + 1):
                    for diag in ("TLBR", "TRBL"):
                        placements.append((r0, c0, h, w, col, diag))
            random.shuffle(placements)
            placements_by_shape.append(placements)

        for rect_a in placements_by_shape[0]:
            cells_a = set(_rect_cells(rect_a))
            for rect_b in placements_by_shape[1]:
                cells_b = set(_rect_cells(rect_b))
                if cells_a & cells_b:
                    continue
                union = cells_a | cells_b
                if (0, 0) not in union or (LEGEND_LATTICE_H - 1, LEGEND_LATTICE_W - 1) not in union:
                    continue
                return [rect_a, rect_b]
    return None


def _render_output(
    size: int,
    bg: int,
    rects: list[tuple[int, int, int, int, int, str]],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> np.ndarray:
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    legend_cells = _all_legend_cells(rects)
    rel_pts = _tile_relative_points(legend_cells, block_size)
    for r, c, col in [_pt for rect in rects for _pt in _rect_markers(rect)]:
        out[lr0 + r * LEGEND_STEP, lc0 + c * LEGEND_STEP] = col
    for rr, cc, col in legend_cells:
        br = base_r + rr * step
        bc = base_c + cc * step
        _paint_fractal_tile(out, br, bc, block_size, rel_pts, col, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size = random.choice(list(BLOCK_SIZE_RANGE))
    step = block_size + BLOCK_GAP

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
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
