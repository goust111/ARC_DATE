# Variant of 3e6067c3
# Dimension: D11 — 空间关系：方块排列拓扑 (meta)
# Change: 方块排列在不规则间距的网格格点上，桥梁仅在行/列对齐的方块之间绘制（正交方向）... → 层级排列（树形结构）

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis
# a) 具体化与行为追踪
#    原题输入里，方块散落在不规则网格点上，底部控制行是一串颜色，solver 每次只处理相邻两色，
#    因而输出天然是一条链。这里把方块改成分层摆放：顶层是父节点，下层是它的子节点，子树整体
#    呈金字塔/层级结构。控制行不再表示“单步跳转序列”，而是表示树的邻接表：
#    [父, 子1, 子2, border, 父, 子1, border, ...]。于是 solver 需要一次读取一个父节点组，
#    然后给该父节点向下一层的全部子节点画出同色分叉桥。一个小例子：
#    原逻辑：控制序列 [2,4,7] => 2->4, 4->7，两条局部链段。
#    新逻辑：控制序列 [2,4,7,border,4,6,border] => 2 同时连到 4 和 7，4 再连到 6。
#    差异在于“下一步”不再唯一，控制行必须恢复一对多关系。
# b) 复杂度校准
#    树只允许 1 或 2 个孩子，模板数量有限，避免自由度过大；同时强制至少一个分叉节点，
#    使输出明显不是链。方块内色全局唯一，避免父/子颜色匹配歧义。
# c) 原 main() 的旧假设审查
#    - aligned(f1,f2) 假设正确桥只发生在同行或同列，这在树形层级里不成立，需要移除。
#    - “当前节点 cur”单指针推进假设链式访问，这在分叉结构里会漏掉兄弟分支，必须改成逐组处理。
#    - draw_bridge(fa,fb,color) 只支持一对一矩形桥；树形需要一个父节点同时生成多条分支，因此改成
#      对一组孩子生成共享竖干 + 横枝 + 竖落点的 T 形/总线式桥。
#    - 原生成器依赖 self-avoiding path，因此每个节点最多两个相邻候选但只访问一个；树形变体里改成
#      先选树模板，再根据子树中心布置父节点。
# d) 放置合法性
#    节点按层放置，列位置由叶子序和子树中心决定，天然不重叠；同一父节点的孩子在下一层连续排列，
#    所以水平枝条只覆盖自己的子树区间。不同层的桥在不同垂直间隙中绘制，不会跨层互撞。
# e) 视觉可观察性
#    人看到样例时，会同时看到：1) 方块呈层级排列；2) 控制行按 border 分块；3) 输出里同色桥从
#    一个父方块分叉到多个子方块。仅靠 2-3 个样例就能看出“分组 = 父连所有子”的规律。
# f) 可视化整洁度
#    桥只画在层间空白区，不穿过任何方块；每层父节点的子树区间互不重叠，避免多色枝条混在一起。
# g) 颜色变化可推断性
#    不引入隐藏颜色映射。桥颜色始终等于父节点内部颜色，和原题“桥由起点色着色”保持一致，只是
#    从一对一推广为一对多。
# h) 变体量化指标
#    关键指标是 branch_edges = sum(max(0, 度数-1))。原题恒为 0；该生成器强制至少一个节点度数为 2，
#    因而 branch_edges >= 1，避免退化回链。
# i) 新 solver 的隐式假设
#    - 控制行使用 border 作为分隔符，generate_input() 始终按该格式构造。
#    - 直接参与控制行的方块内色全局唯一，generator 用不重复颜色保证。
#    - 一个父节点的所有孩子都在下一层，且同层共享同一顶部行；模板布局保证这一点。
#    - 层间至少有 1 行空白让桥落下；生成器固定 row_gap >= 2 保证可绘制。

_DIGITS = [int(i) for i in range(10)]
_TEMPLATES = [
    {0: [1, 2], 1: [3, 4], 2: [5]},
    {0: [1, 2], 1: [3], 2: [4, 5]},
    {0: [1, 2], 1: [3, 4], 2: [5, 6]},
    {0: [1, 2], 1: [3], 2: [4], 4: [5, 6]},
]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, int(bg))
    assert not (border is None)

    frames = _detect_frames(g, int(border))
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    groups = _parse_control_groups(g[int(ctrl_r)], int(bg), int(border))
    out = g.copy()
    for group in groups:
        parent_color = int(group[0])
        child_colors = [int(col) for col in group[1:]]
        parent = _unique_frame(by_color, int(parent_color))
        children = [_unique_frame(by_color, int(col)) for col in child_colors]
        cells = _branch_cells(parent, children)
        for r, c in cells:
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(parent_color)
    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _count in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(g == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        thickness = None
        inner_color = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            colors = set(map(int, inner.flatten()))
            if len(colors) != 1:
                continue
            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue
            thickness = int(tt)
            inner_color = int(next(iter(colors)))
            break

        if thickness is None or inner_color is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(thickness),
                "inner": int(inner_color),
                "inner_bbox": (int(r0 + thickness), int(c0 + thickness), int(r1 - thickness), int(c1 - thickness)),
            }
        )

    assert not (not frames)
    return frames


def _parse_control_groups(row: np.ndarray, bg: int, border: int) -> list[list[int]]:
    tokens = [int(v) for v in row if int(v) != int(bg)]
    groups: list[list[int]] = []
    cur: list[int] = []
    for tok in tokens:
        if int(tok) == int(border):
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(int(tok))
    if cur:
        groups.append(cur)
    assert not (not groups)
    assert all(len(group) >= 2 for group in groups)
    return groups


def _unique_frame(by_color: dict[int, list[dict]], color: int) -> dict:
    assert not (color not in by_color)
    frames = by_color[int(color)]
    assert int(len(frames)) == 1
    return frames[0]


def _paint_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_color: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_color)


def _branch_cells(parent: dict, children: list[dict]) -> set[tuple[int, int]]:
    assert not (not children)
    children = sorted(children, key=lambda frame: int(frame["inner_bbox"][1]))
    pa = parent["bbox"]
    ia = parent["inner_bbox"]
    child_top = int(children[0]["bbox"][0])
    assert all(int(child["bbox"][0]) == int(child_top) for child in children)

    gap_top = int(pa[2] + 1)
    gap_bottom = int(child_top - 1)
    assert int(gap_top) <= int(gap_bottom)

    inner_w = int(ia[3] - ia[1] + 1)
    band_h = int(max(1, min(inner_w, int(gap_bottom - gap_top + 1))))
    band_r0 = int(gap_top + (int(gap_bottom - gap_top + 1) - int(band_h)) // 2)
    band_r1 = int(band_r0 + band_h - 1)

    left = int(min([int(ia[1])] + [int(child["inner_bbox"][1]) for child in children]))
    right = int(max([int(ia[3])] + [int(child["inner_bbox"][3]) for child in children]))

    cells: set[tuple[int, int]] = set()
    for r in range(int(gap_top), int(band_r1) + 1):
        for c in range(int(ia[1]), int(ia[3]) + 1):
            cells.add((int(r), int(c)))
    for r in range(int(band_r0), int(band_r1) + 1):
        for c in range(int(left), int(right) + 1):
            cells.add((int(r), int(c)))
    for child in children:
        ib = child["inner_bbox"]
        for r in range(int(band_r0), int(child["bbox"][0])):
            for c in range(int(ib[1]), int(ib[3]) + 1):
                cells.add((int(r), int(c)))
    return cells


def _node_levels(children_by: dict[int, list[int]]) -> dict[int, int]:
    levels = {0: 0}
    queue = [0]
    for node in queue:
        for child in children_by.get(int(node), []):
            levels[int(child)] = int(levels[int(node)] + 1)
            queue.append(int(child))
    return levels


def _dfs_leaves(node: int, children_by: dict[int, list[int]]) -> list[int]:
    children = children_by.get(int(node), [])
    if not children:
        return [int(node)]
    out: list[int] = []
    for child in children:
        out.extend(_dfs_leaves(int(child), children_by))
    return out


def _desc_leaf_bounds(node: int, children_by: dict[int, list[int]], leaf_index: dict[int, int]) -> tuple[int, int]:
    leaves = _dfs_leaves(int(node), children_by)
    idxs = [int(leaf_index[int(leaf)]) for leaf in leaves]
    return int(min(idxs)), int(max(idxs))


def _internal_order(children_by: dict[int, list[int]], levels: dict[int, int], frames_by_node: dict[int, dict]) -> list[int]:
    nodes = [int(node) for node, children in children_by.items() if children]
    return sorted(nodes, key=lambda node: (int(levels[int(node)]), int(frames_by_node[int(node)]["bbox"][1])))


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]

    children_by = {int(node): [int(ch) for ch in children] for node, children in dict(np.random.choice(_TEMPLATES)).items()}
    nodes = sorted({0} | set(children_by.keys()) | {int(ch) for children in children_by.values() for ch in children})
    if len(nodes) > len(palette):
        return None

    t = 1
    inner = int(np.random.choice(np.array([1, 3], dtype=int)))
    outer = int(inner + 2 * t)
    levels = _node_levels(children_by)
    max_level = max(int(level) for level in levels.values())

    leaf_order = _dfs_leaves(0, children_by)
    leaf_index = {int(node): idx for idx, node in enumerate(leaf_order)}
    slot_gap = int(np.random.choice(np.array([1, 3], dtype=int)))
    slot_step = int(outer + slot_gap)
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    row_gaps = [int(np.random.randint(2, 5)) for _ in range(int(max_level))]

    level_r0 = [int(top_margin)]
    for level in range(int(max_level)):
        level_r0.append(int(level_r0[-1] + outer + row_gaps[int(level)]))

    colors = np.random.permutation(np.array(palette, dtype=int))[: len(nodes)]
    node_color = {int(node): int(col) for node, col in zip(nodes, colors)}

    frames_by_node: dict[int, dict] = {}
    for node in nodes:
        left_leaf, right_leaf = _desc_leaf_bounds(int(node), children_by, leaf_index)
        center_slot = int(left_leaf + right_leaf)
        c0 = int(left_margin + (center_slot * slot_step) // 2)
        r0 = int(level_r0[int(levels[int(node)])])
        frames_by_node[int(node)] = {
            "bbox": (int(r0), int(c0), int(r0 + outer - 1), int(c0 + outer - 1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r0 + outer - t - 1), int(c0 + outer - t - 1)),
            "inner": int(node_color[int(node)]),
            "node": int(node),
        }

    internal_nodes = _internal_order(children_by, levels, frames_by_node)
    ctrl_tokens: list[int] = []
    for node in internal_nodes:
        ctrl_tokens.append(int(node_color[int(node)]))
        ctrl_tokens.extend(int(node_color[int(child)]) for child in children_by[int(node)])
        ctrl_tokens.append(int(border))
    if len(ctrl_tokens) < 3:
        return None

    max_r1 = max(int(frame["bbox"][2]) for frame in frames_by_node.values())
    max_c1 = max(int(frame["bbox"][3]) for frame in frames_by_node.values())
    ctrl_gap = int(np.random.randint(1, 4))
    ctrl_r = int(max_r1 + ctrl_gap)
    ctrl_cols = [int(1 + 2 * i) for i in range(len(ctrl_tokens))]
    right_margin = int(np.random.randint(1, 3))
    H = int(ctrl_r + 2)
    W = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for node in nodes:
        frame = frames_by_node[int(node)]
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _paint_frame(out, r0=int(r0), c0=int(c0), outer=int(outer), t=int(t), border=int(border), inner_color=int(frame["inner"]))

    bridge_cells_all: set[tuple[int, int]] = set()
    branch_metric = 0
    for node in internal_nodes:
        children = [frames_by_node[int(child)] for child in children_by[int(node)]]
        cells = _branch_cells(frames_by_node[int(node)], children)
        if not cells:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(node_color[int(node)])
        bridge_cells_all |= set(cells)
        branch_metric += max(0, int(len(children) - 1))

    if int(branch_metric) < 1:
        return None

    for col, cc in zip(ctrl_tokens, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
