from __future__ import annotations

# Variant of 291dc1e1
# Dimension: D13 — 源-目标基数关系 (meta)
# Change: 多对一——多个矩形合并到一个堆叠输出中（many:1）... → 多对多——按颜色分组输出多个堆叠
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 输入仍然是原任务的几何骨架：一条水平边线和一条垂直边线在某个角点相交，角点上有一个独特标记色；
#   内部散布若干 2xW 的矩形，矩形之间不接触、不碰边线。
# - 原任务把全部矩形按 `(到蓝线距离, 到红线距离, 沿边线离角点的先后)` 排成一个全局序列，然后统一堆成一列。
# - 本变体保留这个全局排序规则，但在“输出组装”阶段增加一级分流：
#   先按矩形的“主色”（bbox 内出现次数最多的颜色）分组，
#   再让每个颜色组内部沿用原任务的排序顺序独立竖直堆叠，
#   最后把这些堆叠块从左到右并列摆放。
# - 小例子：
#   若原全局排序得到 `[A(绿), B(黄), C(绿), D(灰)]`，
#   原输出是 `A/B/C/D` 单列；
#   变体输出改为 3 列：第 1 列堆 `A,C`，第 2 列堆 `B`，第 3 列堆 `D`。
#   若蓝线垂直，则各矩形仍先做 90 度逆时针旋转，再进入各自颜色组。
#
# b) 复杂度校准
# - 仅按颜色分组还不够，如果每个颜色只出现一次，就会退化成“拆成几列但没有真正聚合”。
# - generator 因此强制：
#   1. 至少两个主色组；
#   2. 至少两个主色各自出现不止一次；
#   3. 全局排序序列里颜色要交错，而不是天然按组连续。
# - 这样输出中的“先分组再堆叠”会与原任务的单列排序形成肉眼可见差异。
#
# c) 隐式假设审查（针对原 main 中与变体相关的段落）
# - 原 `rects.sort(...)` 假设所有矩形最终进入同一个 `seq`；变体中这个前提失效，排序只能产生“组内顺序”的基线。
# - 原 `seq.append(a)` / `outW=sum...` / 单个 `out_r` 明确依赖 many:1；变体必须改成“每个颜色组各自维护 seq 和 out_r”。
# - 原代码把整个 bbox 原样作为对象，因此若对象内允许少量异色像素，新的“按颜色分组”不能用“整块颜色完全一致”判断。
#   这里改成按主色分组；generator 也保证主色严格占多数，避免并列。
# - 原始 `blue_orient == "v"` 的旋转逻辑仍然成立，因为旋转发生在分组之前，不依赖 many:1。
#
# d) 放置合法性
# - generator 继续用两条水平带放置矩形：第 1 带在第 2-3 行，第 2 带在第 5-6 行。
# - 同一带中的矩形之间保留 2 列背景间隔；两带之间保留 1 行背景，不重叠。
# - 变换到其他角和蓝线方向时沿用原任务的 `_transform`，因此输入仍满足 solver 对角点、边线和对象分离的假设。
#
# e) 视觉可观察性
# - 输出不再是一整列，而是多个并列列块，这是 D13 变化的直接视觉信号。
# - 每列内部仍保留原任务的距离排序；列与列之间由主色聚合决定。
# - 为了让“同色归并”明显，generator 让同色矩形在输入中分散出现，而不是集中相邻。
#
# f) 可视化整洁度
# - 主色只从 6 个非边线颜色中选；异色点仅作少量点缀，不会破坏矩形轮廓。
# - 输出各堆叠块之间固定插入 1 列背景，避免不同颜色列粘连。
#
# g) 颜色变化可推断性
# - 这里没有颜色映射，颜色本身就是分组键。
# - 由于对象内部允许极少数边线色噪点，代码和 generator 都按“主色”理解颜色组；
#   人类也能从大面积主填充色直观看出同组关系。

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = g.shape
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            return int(r), int(c)
    raise AssertionError("corner not found")


def _dominant_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _count_column_runs(grid: np.ndarray, bg: int) -> int:
    active = np.any(grid != int(bg), axis=0)
    runs = 0
    prev = False
    for flag in active:
        cur = bool(flag)
        if cur and not prev:
            runs += 1
        prev = cur
    return int(runs)


def _has_repeat_support(counts: dict[int, int]) -> bool:
    return len([v for v in counts.values() if int(v) >= 2]) >= 2


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc = _locate_corner(g, bg)
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy(), "group": _dominant_color(sub)})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rct in rects:
        rct["db"] = int(dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(dist_to_edge(rct, red_orient, red_edge_idx))

    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            along_from_corner(r, blue_orient),
            along_from_corner(r, red_orient),
        )
    )

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        grp = int(rct["group"])
        if grp not in groups:
            groups[grp] = []
            group_order.append(grp)
        groups[grp].append(a)

    blocks = []
    for grp in group_order:
        seq = groups[grp]
        block_w = int(max(int(a.shape[1]) for a in seq))
        block_h = int(sum(int(a.shape[0]) for a in seq))
        block = np.full((block_h, block_w), int(bg), dtype=int)
        out_r = 0
        for a in seq:
            h, w = map(int, a.shape)
            out_c = int((block_w - w) // 2)
            paste(block, a, int(out_r), int(out_c))
            out_r += int(h)
        blocks.append(block)

    out_h = int(max(int(block.shape[0]) for block in blocks))
    out_w = int(sum(int(block.shape[1]) for block in blocks) + max(0, len(blocks) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_c = 0
    for block in blocks:
        h, w = map(int, block.shape)
        paste(out, block, 0, int(out_c))
        out_c += int(w) + 1

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([BLUE, RED], dtype=int)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    max_minor = min(3, max(0, (area - 1) // 3))
    if max_minor == 0 or np.random.rand() < 0.45:
        return patch
    n_minor = int(np.random.randint(1, max_minor + 1))
    cells = np.random.choice(area, size=n_minor, replace=False)
    colors = np.random.choice(MINORITY_COLORS, size=n_minor, replace=True)
    flat = patch.reshape(-1)
    for cell, color in zip(cells, colors):
        if int(color) != int(dominant):
            flat[int(cell)] = int(color)
    return patch


def _sample_band_widths(count: int, long_dim: int) -> list[int]:
    used_lo = 4 * count - 2
    used_hi = min(long_dim - 3, 10 * count - 2)
    choices = np.arange(used_lo, used_hi + 1, 2)
    assert len(choices) > 0
    used_target = int(np.random.choice(choices))
    return _distribute_band_widths(count, used_target)


def _run_count(seq: list[int]) -> int:
    runs = 0
    prev = None
    for color in seq:
        if color != prev:
            runs += 1
            prev = color
    return int(runs)


def _build_sequence(n0: int, n1: int) -> list[int]:
    total = n0 + n1
    n_groups = 2 if total <= 5 or np.random.rand() < 0.55 else 3
    colors = [int(c) for c in np.random.choice(DOMINANT_COLORS, size=n_groups, replace=False)]

    if n_groups == 2:
        counts = [2, 2]
    else:
        counts = [2, 2, 1]
    remaining = total - sum(counts)
    assert remaining >= 0
    for _ in range(remaining):
        counts[int(np.random.randint(n_groups))] += 1

    bag = []
    for color, count in zip(colors, counts):
        bag.extend([int(color)] * int(count))

    for _ in range(200):
        seq = [int(x) for x in np.random.permutation(bag)]
        top = seq[:n0]
        bot = seq[n0:]
        if len(set(seq)) < 2:
            continue
        if len(set(top)) < 2 or len(set(bot)) < 2:
            continue
        if not any((color in top) and (color in bot) for color in set(seq)):
            continue
        if _run_count(seq) <= len(set(seq)):
            continue
        return seq
    raise RuntimeError("failed to build interleaved dominant-color sequence")


def _build_canonical() -> tuple[np.ndarray, dict]:
    long_dim = int(np.random.randint(15, 29))
    max_count = min(4, (long_dim - 1) // 4)
    pairs = [(a, b) for a in range(2, max_count + 1) for b in range(2, max_count + 1) if 5 <= a + b <= 7]
    if not pairs:
        raise RuntimeError("no feasible band counts")

    n0, n1 = pairs[int(np.random.randint(len(pairs)))]
    band0 = _sample_band_widths(n0, long_dim)
    band1 = _sample_band_widths(n1, long_dim)
    seq = _build_sequence(n0, n1)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    placements = []
    idx = 0
    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            dominant = int(seq[idx])
            patch = np.full((2, int(w)), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(r0), int(c))
            placements.append({"dominant": dominant, "row": int(r0), "col": int(c), "width": int(w)})
            c += int(w) + 2
            idx += 1

    counts = {color: seq.count(color) for color in sorted(set(seq))}
    return grid, {"sequence": seq, "counts": counts, "n_groups": len(counts), "placements": placements}


def _gen_attempt() -> tuple[np.ndarray, dict]:
    canonical, info = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    info = dict(info)
    info["blue_orient"] = blue_orient
    info["corner"] = corner
    return _transform(canonical, blue_orient, corner), info


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid, info = _gen_attempt()
        if int(info["n_groups"]) < 2:
            continue
        if not _has_repeat_support(info["counts"]):
            continue
        out = main(grid)
        if _count_column_runs(out, int(get_bg(grid))) != int(info["n_groups"]):
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D13 variant")
