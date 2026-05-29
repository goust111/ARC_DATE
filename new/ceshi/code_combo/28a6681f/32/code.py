# Combo variant of 28a6681f (index 32)
# combo_spec: D5/0+D6/2+D8/1
# Name: 矩形水块+高度编码填充+多盆地

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 保持 10x10、背景 0、水 1 的原题视觉词汇，但把输入水源改成一个规则矩形块，且总是贴着左边
#   或右边边界放在上方平台上。
# - 下半区改成 4 或 5 个小盆地链。为避免 10x10 中横向空间不够，盆地都做成 1 列宽窄井，
#   相邻井之间由底部锚定的竖墙隔开；4 盆地模板额外留出 2 列空白死区，5 盆地模板则占满整条链。
# - 输出不再追求“最深稳定位置”。若源矩形高为 h，则每个盆地的目标是“本盆地最底部 h 层”。
#   例如左侧 3x4 水块输入到 5 个一列宽盆地时，总水量是 12。组合规则下，前 4 个近盆地各堆成
#   3 层，最远的第 5 个盆地为空；原题局部物理则会优先把靠近入口的盆地灌得更深，因此同一输入
#   会得到明显不同的输出。
#
# b) 复杂度校准
# - 关键自由度只保留 3 个：源矩形高度 h in {2,3}、源矩形宽度 w、盆地数 in {4,5}。
#   h 控制每个盆地的目标层数，w* h 控制最多能灌满多少个盆地。
# - 盆地都做成 1 列宽，避免在“水量不足”时出现难以读懂的锯齿状半满容器；这样解题者能直接
#   看出“高度编码层数 + 水量决定填满几个前缀盆地”的组合规律。
# - 这里不引入 D5/0 单变体文件里的外部水位标记，因为 combo_spec 对 D5/0 的权威描述是
#   “水源对象形态改为矩形块”，不是“目标由标记线决定”。
#
# c) 对原 main() 的隐式假设审查
# - 原 `settle()` 假设答案完全由像素级重力和滑动决定。融合后目标由源矩形高度编码，因此逐像素
#   模拟不再适用，必须改成“读源矩形 -> 读盆地链 -> 计数分配”。
# - 原 `check_output` 要求每个水像素左/下/右都被挡住；高度编码后的顶部水层会暴露侧面，所以该
#   断言不再成立，改为检查“所有输出水都落在各盆地预定义的底部 h 层目标格中”。
# - 原代码默认底部结构只需支撑最终物理稳定性。这里新增假设：底行非背景列就是盆地分隔墙，并且
#   盆地链从源侧边界开始，到最远的最后一根墙结束。生成器专门按这个几何族构造，避免歧义。
#
# d) 放置合法性
# - 生成器固定使用经过筛选的 4/5 盆地模板：所有墙柱都底部锚定、不重叠、不越界。
# - 源矩形底下总有一条上方平台支撑，所以输入水全部合法；平台不触底，不会干扰盆地解析。
# - 上半区额外只放 2 个小挡块，它们从不接触底行，因此不会被误识别成盆地墙，也不会占用目标水层。
#
# e) 视觉可观察性
# - 输入中矩形源的“高度”直接可见，输出中每个被触达盆地的堆叠高度与之完全一致。
# - 当 w 小于盆地数时，只会看到靠近源侧的一串前缀盆地被填到同一高度，远处盆地保持空白；
#   这把“高度编码”和“近盆地优先分配”同时暴露出来。
# - 左右镜像、4/5 盆地两种家族、以及多组颜色复用提供重复支撑，但不引入额外隐藏规则。
#
# f) 可视化整洁度
# - 结构只用 3 种角色：水、上方平台/挡块、下方墙体。挡块数量严格限制为 2 个，且都位于中上部，
#   不会和盆地填充区域叠在一起。
# - 窄盆地链虽然规则，但通过 4/5 两种数量、左右镜像和交替墙色保持了足够变化，不会退化成
#   “几乎每次都同一张图”。
#
# g) 变化的可推断性
# - 所有行为只依赖可见几何关系：源矩形的高和宽、盆地从哪一侧开始、以及盆地数量。
# - 颜色不携带排序、匹配或数值含义；颜色只分“水”“上方结构”“下方结构”三类语义。
#
# h) 量化退化检查目标
# - 关键指标 1：每个输出样本中，所有非空盆地的水高都必须等于源矩形高度。
# - 关键指标 2：被填满的盆地必须组成从源侧开始的连续前缀，且前缀长度由总水量决定。
# - 关键指标 3：对同一输入运行原题经典物理，输出应与本变体不同。生成器用内置 `_classic_main`
#   过滤退化模板。
#
# i) main() 新增假设与 generator 保证
# - `main()` 假设所有水都属于同一个贴边矩形源，且它贴住上边界和单侧边界；生成器严格如此放置。
# - `main()` 假设底行非背景非水列构成 4 或 5 根分隔墙；生成器只用这一几何族。
# - `main()` 假设每个盆地列至少有 h 个最深背景格可填；模板墙高专门选在 5/6 行，保证 h in {2,3}
#   时底部 h 层始终存在。


BG = 0
WATER = 1
H = 10
W = 10
SHELF_WIDTH = 5
STRUCTURE_PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)

TEMPLATES = (
    {
        "name": "left_five",
        "side": "left",
        "walls": (1, 3, 5, 7, 9),
        "tops": (5, 6, 5, 6, 5),
        "blockers": ((4, 4), (4, 5)),
        "shelf": (0, 4),
        "source_widths": (3, 4, 5),
    },
    {
        "name": "right_five",
        "side": "right",
        "walls": (0, 2, 4, 6, 8),
        "tops": (5, 6, 5, 6, 5),
        "blockers": ((4, 1), (4, 3)),
        "shelf": (5, 9),
        "source_widths": (3, 4, 5),
    },
    {
        "name": "left_four",
        "side": "left",
        "walls": (1, 3, 5, 7),
        "tops": (5, 6, 5, 6),
        "blockers": ((4, 0), (4, 4)),
        "shelf": (0, 4),
        "source_widths": (3, 4),
    },
    {
        "name": "right_four",
        "side": "right",
        "walls": (2, 4, 6, 8),
        "tops": (6, 5, 6, 5),
        "blockers": ((4, 0), (4, 3)),
        "shelf": (5, 9),
        "source_widths": (3, 4),
    },
)


def _component_count(points: set[tuple[int, int]]) -> int:
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _source_box(g: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == int(WATER))
    assert pts.size > 0, "expected a non-empty source rectangle"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _source_info(g: np.ndarray) -> tuple[int, int, int, int, int, int, str]:
    r0, c0, r1, c1 = _source_box(g)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert r0 == 0, "source rectangle must touch the top edge"
    if c0 == 0 and c1 < g.shape[1] - 1:
        side = "left"
    elif c1 == g.shape[1] - 1 and c0 > 0:
        side = "right"
    else:
        raise AssertionError("source rectangle must touch exactly one side edge")
    return height, width, r0, c0, r1, c1, side


def _parse_basins(g: np.ndarray, side: str) -> list[list[int]]:
    wall_cols = [int(c) for c in range(g.shape[1]) if int(g[g.shape[0] - 1, c]) not in (BG, WATER)]
    assert 4 <= len(wall_cols) <= 5, f"expected 4-5 divider walls, got {wall_cols}"
    basins: list[list[int]] = []
    if side == "left":
        prev = -1
        for wall in sorted(wall_cols):
            cols = [int(c) for c in range(prev + 1, wall)]
            assert cols, f"empty basin before wall {wall}"
            basins.append(cols)
            prev = int(wall)
    else:
        prev = g.shape[1]
        for wall in sorted(wall_cols, reverse=True):
            cols = [int(c) for c in range(prev - 1, wall, -1)]
            assert cols, f"empty basin before wall {wall}"
            basins.append(cols)
            prev = int(wall)
    assert 4 <= len(basins) <= 5, f"expected 4-5 basins, got {len(basins)}"
    return basins


def _basin_target_cells(base: np.ndarray, cols: list[int], height: int) -> list[tuple[int, int]]:
    target_set: set[tuple[int, int]] = set()
    for c in cols:
        empties = [int(r) for r in range(base.shape[0] - 1, -1, -1) if int(base[r, c]) == int(BG)]
        assert len(empties) >= int(height), f"column {c} lacks {height} empty cells"
        for r in empties[:height]:
            target_set.add((int(r), int(c)))

    ordered: list[tuple[int, int]] = []
    for r in range(base.shape[0] - 1, -1, -1):
        for c in cols:
            cell = (int(r), int(c))
            if cell in target_set:
                ordered.append(cell)
    assert len(ordered) == len(cols) * int(height)
    return ordered


def _fill_plan(base: np.ndarray, basins: list[list[int]], height: int, total_water: int) -> list[tuple[int, int]]:
    remaining = int(total_water)
    plan: list[tuple[int, int]] = []
    for cols in basins:
        if remaining <= 0:
            break
        targets = _basin_target_cells(base, cols, height)
        take = min(remaining, len(targets))
        plan.extend(targets[:take])
        remaining -= int(take)
    assert remaining == 0, "not enough basin capacity for the encoded water amount"
    return plan


def _apply_plan(base: np.ndarray, plan: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for r, c in plan:
        assert int(out[r, c]) == int(BG), f"fill plan hit a non-bg cell at {(r, c)}"
        out[r, c] = int(WATER)
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redistribute a rectangular water source into a chain of small basins, using the source height as the destination layer count.

    Input structure:
    - bg = 0 and water = 1.
    - The only water object is one filled rectangle touching the top edge and exactly one side edge.
    - Bottom-anchored divider columns create 4 or 5 narrow basins along the lower edge; a short upper platform supports the source,
      and a couple of small upper blockers may appear to break the original physics symmetry.

    Rule:
    - Read the source rectangle height h and total water count.
    - Remove the source water, parse the basin chain starting from the same side, and for each basin mark its deepest h cells per column
      as that basin's target layer set.
    - Fill basins in source-to-far order. A basin receives at most its h-layer target capacity; if water runs out early, only the
      nearest prefix of basins is filled, with the final touched basin filled bottom-up within its own target cells.
    - All non-water structure pixels stay unchanged.

    Key insight:
    - The rectangle's visible height encodes destination height, while its total area controls how many nearby basins can reach that
      height. The answer is therefore a height-coded prefix fill, not a local deepest-resting simulation.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), f"expected 10x10 grid, got {g.shape}"
    total_water = int(np.sum(g == int(WATER)))
    assert total_water > 0, "input needs source water"

    height, width, r0, c0, r1, c1, side = _source_info(g)
    basins = _parse_basins(g, side)

    if check_input:
        source = {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}
        assert _component_count(source) == 1, "source water must be one 4-connected component"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(WATER)), "source water must be a filled rectangle"
        assert total_water == int(height * width), "all water must belong to the single source rectangle"
        assert 2 <= int(height) <= 3, "generator uses source heights 2 or 3"
        assert 3 <= int(width) <= int(SHELF_WIDTH), "generator uses a compact source width between 3 and 5"
        for r, c in source:
            assert r == H - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"
            left_bg = c > 0 and int(g[r, c - 1]) == int(BG)
            right_bg = c + 1 < W and int(g[r, c + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    base = g.copy()
    base[base == int(WATER)] = int(BG)
    plan = _fill_plan(base, basins, height, total_water)
    out = _apply_plan(base, plan)

    if check_output:
        assert int(np.sum(out == int(WATER))) == int(total_water), "water count must be conserved"
        assert not np.any(out[r0 : r1 + 1, c0 : c1 + 1] == int(WATER)), "source water must leave the source rectangle"
        allowed = set()
        for cols in basins:
            allowed.update(_basin_target_cells(base, cols, height))
        for r, c in np.argwhere(out == int(WATER)):
            assert (int(r), int(c)) in allowed, "output water must stay inside the h-layer basin targets"
        expected = _apply_plan(base, _fill_plan(base, basins, height, total_water))
        assert np.array_equal(out, expected), "output must match the deterministic prefix-fill rule"

    return out


def _classic_main(grid: np.ndarray) -> np.ndarray:
    """Original deepest-resting solver, kept only to reject degenerate generator samples."""
    g = np.array(grid, dtype=int).copy()
    H0, W0 = g.shape

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H0 * W0):
            if rr + 1 < H0 and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H0
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H0
                and cc + 1 < W0
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                if len(opts) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {opts}")
                rr, cc = opts[0]
                continue

            if rr + 1 < H0 and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W0 and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                if best == int(cc):
                    raise AssertionError("classic settle found a zero-step move")
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                if int(g[rr, nx]) != int(BG):
                    raise AssertionError("classic horizontal slide hit a blocked cell")
                cc = nx
                continue

            break
        else:
            raise AssertionError("classic water pixel did not settle")
        return int(rr), int(cc)

    for _ in range(H0 * W0):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            return g
    raise AssertionError("classic water simulation did not converge")


def _weighted_source_width(rng: random.Random, template: dict[str, object]) -> int:
    widths = list(template["source_widths"])
    full_width = int(widths[-1])
    if rng.random() < 0.7:
        partials = [int(w) for w in widths if int(w) != full_width]
        if partials:
            return int(rng.choice(partials))
    return int(rng.choice(widths))


def _draw_template(
    template: dict[str, object],
    height: int,
    source_width: int,
    colors: tuple[int, int, int],
    include_water: bool,
) -> np.ndarray:
    g = np.zeros((H, W), dtype=int)
    shelf_color, wall_a, wall_b = colors

    for idx, (wall, top) in enumerate(zip(template["walls"], template["tops"])):
        color = int(wall_a if idx % 2 == 0 else wall_b)
        g[int(top) :, int(wall)] = color

    for r, c in template["blockers"]:
        g[int(r), int(c)] = int(shelf_color)

    shelf_start, shelf_end = template["shelf"]
    g[int(height), int(shelf_start) : int(shelf_end) + 1] = int(shelf_color)

    if include_water:
        if template["side"] == "left":
            g[: int(height), : int(source_width)] = int(WATER)
        else:
            g[: int(height), W - int(source_width) :] = int(WATER)

    return g


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    template = dict(rng.choice(TEMPLATES))
    height = int(rng.choice((2, 3)))
    source_width = _weighted_source_width(rng, template)
    colors = tuple(int(x) for x in rng.sample(list(STRUCTURE_PALETTE), 3))
    inp = _draw_template(template, height, source_width, colors, include_water=True)
    base = _draw_template(template, height, source_width, colors, include_water=False)

    basins = _parse_basins(base, str(template["side"]))
    plan = _fill_plan(base, basins, height, int(height * source_width))
    out = _apply_plan(base, plan)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        inp, expected = _build_case(rng)
        try:
            solved = main(inp)
            classic = _classic_main(inp)
        except Exception:
            continue
        if np.array_equal(solved, expected) and not np.array_equal(classic, expected):
            return inp
    raise RuntimeError("failed to generate a non-degenerate combo variant sample")
