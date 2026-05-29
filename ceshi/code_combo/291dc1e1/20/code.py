# Combo variant of 291dc1e1 (index 20)
# combo_spec: D12/1+D13/0
# Name: 角落编码方向+分色多输出

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题输入仍保留：蓝线与红线在一个角落相交，角点是单像素标记，内部是若干 2xW 的实心矩形。
# - 原题先按 `(到蓝线距离 db, 到红线距离 dr, 沿蓝线离角点先后, 沿红线离角点先后)` 排出一个全局序列，
#   若蓝线垂直则每块先逆时针旋转 90 度，然后所有块统一自上而下堆成一列。
# - 本融合变体保留“检测矩形、按两条边线距离排序、蓝线垂直则统一逆时针旋转”这条主链；
#   改动发生在输出组装：
#   1. 先按矩形主色分组，同色矩形共享一个输出组；
#   2. 再由角点位置决定每个组内部的堆叠方向：
#      `tl -> 上到下`, `br -> 下到上`, `bl -> 左到右`, `tr -> 右到左`；
#   3. 各颜色组都使用同一个角落语义，并沿其正交轴并排摆放。
# - 具体小例子：
#   全局排序若得到 `[A(绿,w=2), B(黄,w=4), C(绿,w=6), D(黄,w=2)]`。
#   原题输出是单列 `A/B/C/D`。
#   本题若角在左下，则方向是左到右：
#   绿组先取 `[A,C]`，黄组取 `[B,D]`，分别做两条水平拼接行，再把两行自上而下排开。
#   若角在右下，则改成下到上：每组仍是竖向堆叠，但同组内部顺序上下翻转，且各组整体贴齐底边。
# - 差异因此是“颜色先分流 + 角落决定每组内部朝向”，两项共同决定输出布局。
#
# b) 复杂度校准
# - 只做按色分组容易退化成“多列但方向没作用”；只做角落方向又可能退化成“仍然只有一列”。
# - generator 因此强制：
#   1. 至少两个颜色组；
#   2. 至少两个颜色各自重复出现；
#   3. 颜色在全局排序里交错，而不是天然成团；
#   4. 至少一个重复颜色组包含不同宽度的矩形，避免正反方向碰巧等价。
# - 方向档位只保留四种，与四个角一一对应，足够明显且不会失控。
#
# c) 隐式假设审查（针对原 main 相关段落）
# - 原 `rects.sort(...)` 之后默认把所有对象送进同一个 `seq`。D13 变化打破了 many:1 前提，
#   现在排序只定义“组内先后”，之后必须按主色拆成多个序列。
# - 原 `outW=max(...)`, `outH=sum(...)`, `out_r+=h` 只适用于“统一自上而下堆一列”；
#   D12/1 后需要四套堆叠器：上到下、下到上、左到右、右到左。
# - 原代码把 `(g != marker)` 写进对象 mask。那隐含“角点颜色绝不会出现在矩形内”。
#   本融合虽然仍让 marker=0 且生成器不在矩形里用 0，但 solver 只去掉真实角点单元，
#   不依赖“全网格所有 0 都不是对象”这条更强假设。
# - 原 `blue_orient == "v"` 触发统一逆时针旋转的逻辑仍然成立，因为它与输出分组、输出朝向正交。
#
# d) 放置合法性
# - 生成器仍用原题的两条水平带：第 1 带在 canonical 的第 2-3 行，第 2 带在第 5-6 行。
# - 同带矩形之间固定留 2 列背景，两个带之间固定留 1 行背景，因此对象不会接触或重叠。
# - 先在 canonical 的左上角蓝横红竖框架内生成，再用整图翻转/旋转搬到任意角与蓝线方向，合法性保持不变。
#
# e) 视觉可观察性
# - 人类从少量样例能看到两层稳定规律：
#   1. 同色矩形最终被聚到同一个输出块中；
#   2. 角点从不同角落移动时，所有颜色组内部的排布朝向一起改变。
# - 左右方向与上下方向会直接改变输出的长宽主轴；正反方向则通过同组内不同宽度矩形的前后次序体现。
#
# f) 可视化整洁度
# - 颜色组之间只留 1 格背景间隔，既能分开，又不会把输出拉得过散。
# - 主色只从 `{3,4,5,6,7,9}` 取；异色噪点只用边线色 `{1,2}`，角点 0 保持孤立。
# - 对象仍是清晰的轴对齐矩形，没有叠盖标记，没有多层编码压在同一区域。
#
# g) 颜色变化可推断性
# - 这里没有新的颜色映射，颜色本身就是分组键。
# - 因为允许少量异色像素，所以 solver 与 generator 都按“主色”理解一块属于哪个组；
#   generator 保证主色严格占多数，避免分组歧义。
#
# h) 融合行为量化与防退化策略
# - 生成器内部会对候选样本做两种对照：
#   1. 与“同方向但不分组”的 many:1 输出比较，若相同则说明 D13 退化，丢弃；
#   2. 与“已分组但取该轴反方向”的输出比较，若相同则说明 D12 的方向对该样本不起作用，丢弃。
# - 因此保留下来的样本同时需要让“按色分流”与“角落方向”都真实改变输出。
#
# i) 新 solver 假设与 generator 保证
# - 假设：角点一定在四角之一，且相邻两格是两条边线。生成器始终如此构造。
# - 假设：内区对象都是 bbox 内无背景的 2xW 实心矩形。生成器始终如此构造，并只加少量非背景噪点。
# - 假设：同色分组的主色唯一。生成器把噪点数量限制在面积的不到一半，保证主色严格多数。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([BLUE, RED], dtype=int)

DIR_BY_CORNER = {
    "tl": "top_to_bottom",
    "tr": "right_to_left",
    "bl": "left_to_right",
    "br": "bottom_to_top",
}
OPPOSITE_DIR = {
    "top_to_bottom": "bottom_to_top",
    "bottom_to_top": "top_to_bottom",
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
}


def _corner_key(cr: int, cc: int, H: int, W: int) -> str:
    return ("t" if int(cr) == 0 else "b") + ("l" if int(cc) == 0 else "r")


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


def _extract_sorted_rects(g: np.ndarray) -> tuple[list[dict], int, str, str, str]:
    H, W = g.shape
    bg = int(get_bg(g))
    cr, cc = _locate_corner(g, bg)
    corner_key = _corner_key(cr, cc, H, W)
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
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

    mask = (g != int(bg)) & (~line_mask)
    mask[cr, cc] = False

    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, b)),
                "arr": sub.copy(),
                "group": _dominant_color(sub),
                "marker": marker,
            }
        )

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

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )

    return rects, bg, blue_orient, red_orient, corner_key


def _build_block(seq: list[np.ndarray], bg: int, stack_dir: str) -> np.ndarray:
    assert seq
    if stack_dir in ("top_to_bottom", "bottom_to_top"):
        block_h = int(sum(int(arr.shape[0]) for arr in seq))
        block_w = int(max(int(arr.shape[1]) for arr in seq))
        block = np.full((block_h, block_w), int(bg), dtype=int)
        if stack_dir == "top_to_bottom":
            out_r = 0
            for arr in seq:
                h, w = map(int, arr.shape)
                out_c = int((block_w - w) // 2)
                paste(block, arr, int(out_r), int(out_c))
                out_r += int(h)
        else:
            out_r = int(block_h)
            for arr in seq:
                h, w = map(int, arr.shape)
                out_r -= int(h)
                out_c = int((block_w - w) // 2)
                paste(block, arr, int(out_r), int(out_c))
        return block

    block_h = int(max(int(arr.shape[0]) for arr in seq))
    block_w = int(sum(int(arr.shape[1]) for arr in seq))
    block = np.full((block_h, block_w), int(bg), dtype=int)
    if stack_dir == "left_to_right":
        out_c = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            out_r = int((block_h - h) // 2)
            paste(block, arr, int(out_r), int(out_c))
            out_c += int(w)
    else:
        out_c = int(block_w)
        for arr in seq:
            h, w = map(int, arr.shape)
            out_c -= int(w)
            out_r = int((block_h - h) // 2)
            paste(block, arr, int(out_r), int(out_c))
    return block


def _assemble_blocks(blocks: list[np.ndarray], bg: int, stack_dir: str) -> np.ndarray:
    assert blocks
    if stack_dir in ("top_to_bottom", "bottom_to_top"):
        out_h = int(max(int(block.shape[0]) for block in blocks))
        out_w = int(sum(int(block.shape[1]) for block in blocks) + max(0, len(blocks) - 1))
        out = np.full((out_h, out_w), int(bg), dtype=int)
        out_c = 0
        for block in blocks:
            h, w = map(int, block.shape)
            out_r = 0 if stack_dir == "top_to_bottom" else int(out_h - h)
            paste(out, block, int(out_r), int(out_c))
            out_c += int(w) + 1
        return out

    out_h = int(sum(int(block.shape[0]) for block in blocks) + max(0, len(blocks) - 1))
    out_w = int(max(int(block.shape[1]) for block in blocks))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for block in blocks:
        h, w = map(int, block.shape)
        out_c = 0 if stack_dir == "left_to_right" else int(out_w - w)
        paste(out, block, int(out_r), int(out_c))
        out_r += int(h) + 1
    return out


def _solve(grid: np.ndarray, *, override_dir: str | None = None, group_by_color: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int)
    rects, bg, blue_orient, _red_orient, corner_key = _extract_sorted_rects(g)
    stack_dir = str(override_dir if override_dir is not None else DIR_BY_CORNER[corner_key])

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    flat_seq: list[np.ndarray] = []

    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        flat_seq.append(arr)
        grp = int(rect["group"])
        if grp not in groups:
            groups[grp] = []
            group_order.append(grp)
        groups[grp].append(arr)

    if not group_by_color:
        return _build_block(flat_seq, bg, stack_dir)

    blocks = [_build_block(groups[grp], bg, stack_dir) for grp in group_order]
    return _assemble_blocks(blocks, bg, stack_dir)


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, group_by_color=True)


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
    return [2 + 2 * int(unit) for unit in units]


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
        if int(color) != prev:
            runs += 1
            prev = int(color)
    return int(runs)


def _has_repeat_support(counts: dict[int, int]) -> bool:
    return len([count for count in counts.values() if int(count) >= 2]) >= 2


def _build_sequence(n0: int, n1: int) -> list[int]:
    total = int(n0 + n1)
    n_groups = 2 if total <= 5 or np.random.rand() < 0.55 else 3
    colors = [int(c) for c in np.random.choice(DOMINANT_COLORS, size=n_groups, replace=False)]

    counts = [2, 2] if n_groups == 2 else [2, 2, 1]
    remaining = int(total - sum(counts))
    assert remaining >= 0
    for _ in range(remaining):
        counts[int(np.random.randint(n_groups))] += 1

    bag = []
    for color, count in zip(colors, counts):
        bag.extend([int(color)] * int(count))

    for _ in range(300):
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


def _repeat_group_has_width_variation(placements: list[dict], counts: dict[int, int]) -> bool:
    for color, count in counts.items():
        if int(count) < 2:
            continue
        widths = {int(p["width"]) for p in placements if int(p["dominant"]) == int(color)}
        if len(widths) >= 2:
            return True
    return False


def _count_runs_along_axis(grid: np.ndarray, bg: int, axis: str) -> int:
    if axis == "col":
        active = np.any(grid != int(bg), axis=0)
    else:
        active = np.any(grid != int(bg), axis=1)
    runs = 0
    prev = False
    for flag in active:
        cur = bool(flag)
        if cur and not prev:
            runs += 1
        prev = cur
    return int(runs)


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
        for width in widths:
            dominant = int(seq[idx])
            patch = np.full((2, int(width)), dominant, dtype=int)
            patch = _sprinkle_minorities(patch, dominant)
            paste(grid, patch, int(r0), int(c))
            placements.append({"dominant": dominant, "row": int(r0), "col": int(c), "width": int(width)})
            c += int(width) + 2
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
    info["stack_dir"] = DIR_BY_CORNER[corner]
    return _transform(canonical, blue_orient, corner), info


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, info = _gen_attempt()
        if int(info["n_groups"]) < 2:
            continue
        if not _has_repeat_support(info["counts"]):
            continue
        if not _repeat_group_has_width_variation(info["placements"], info["counts"]):
            continue

        stack_dir = str(info["stack_dir"])
        bg = int(get_bg(grid))
        out = main(grid)
        flat_out = _solve(grid, override_dir=stack_dir, group_by_color=False)
        opposite_out = _solve(grid, override_dir=OPPOSITE_DIR[stack_dir], group_by_color=True)

        if np.array_equal(out, flat_out):
            continue
        if np.array_equal(out, opposite_out):
            continue

        axis = "col" if stack_dir in ("top_to_bottom", "bottom_to_top") else "row"
        if _count_runs_along_axis(out, bg, axis) != int(info["n_groups"]):
            continue

        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
