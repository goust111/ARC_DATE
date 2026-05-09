# Variant of 20270e3b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 两个精灵的连接器具有相同的归一化形状（相同长度的ctrl色水平条），按形状/颜色匹配... → 边缘行列对齐匹配

from __future__ import annotations

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# 设计分析（先分析，再编码）
# ---------------------------------------------------------------------------
# 参考任务 78332cb0 的 solver 源码在当前工作区里不可用；这里只能依据：
# 1) 该题的 evaluation JSON 样例；
# 2) 仓库里的机制标注：
#    "s.bottom-edge fg cols == d.top-edge fg cols"。
# 因此本变体采用最直接、最可观察的 edge-match 版本：base 的底边前景列模式
# 与 plug 的顶边前景列模式一致，随后把 plug 平移到 base 正下方一行。
#
# a) 具体化与行为追踪
# - 原题输入里有两个 sprite，各自带 ctrl 色水平连接器。solver 先找 ctrl 条，
#   验证两条 connector 的归一化形状一致，再把 plug 平移到 base 上方一行进行对接。
# - 本变体取消“用 connector 决定对接”的推理。输入里只保留普通 obj 前景；
#   两个 sprite 的关键特征变成外轮廓边缘：
#   - base: 看其 bbox 最底行有哪些前景列；
#   - plug: 看其 bbox 最顶行有哪些前景列；
#   - 若两者在各自 bbox 坐标系中的列集合完全一致，就说明可以像瓷砖一样上下拼接。
# - 具体小例子：
#   原逻辑：base connector = row 3 cols {2,3}, plug connector = row 0 cols {2,3}，
#   因而用 connector 左端对齐，plug 被放到 base 上方。
#   新逻辑：base 底边 = row 4 cols {0,2,3}，plug 顶边 = row 0 cols {0,2,3}，
#   因而按 bbox 左边对齐，把 plug 放到 base 下方；真正参与匹配的是边缘稀疏模式，
#   而不是某个特殊颜色条。
#
# b) 复杂度校准
# - 如果底边/顶边总是整段实心横条，那么视觉上会退化回“像 connector 一样的横线”。
#   所以 generator 强制 seam pattern 至少有一个 gap，例如 {0,2,3}、{1,3}。
# - seam 列数限制在 2-4，sprite 宽度限制在 4-9，避免自由度过高导致稀疏到不可读，
#   同时又足够区别于原题的 1-3 长连续 connector。
#
# c) 原 main() 的隐式假设审查
# - `conn(sprite)` / `norm(conn_pts)` / `assert shpA == shpB)`：
#   这些都依赖“存在 ctrl connector 且用它决定匹配”，在变体中不再成立，必须删除。
# - `base_body = base - connD` / `plug_body = plug - connP` 以及上下自由侧检查：
#   依赖 connector 是 sprite 内部一条特殊横条；边缘匹配里没有这条结构，必须删除。
# - `tr = (dr0 - 1) - pr0`：
#   原式把 plug 放到 base 上方，并按 connector 行对齐。变体应改成
#   `tr = base_bbox_bottom + 1 - plug_bbox_top`，即上下拼接。
# - overlap 规则：
#   原题允许 ctrl 与 obj 在 connector 区域重叠；新变体不允许任何重叠，
#   因为两块是边对边相邻而不是共享一行。
#
# d) 放置合法性
# - generator 先构造“已拼好”的 seam 关系，再裁成两个 sprite，最后把 plug 散放到画布别处。
# - base 固定贴在输入左上角 `(0,0)`，plug 用 `random_free_location(..., padding=1)` 放置，
#   保证两块既不重叠，也不会 8 邻接粘成一个组件。
# - solver 假设输出紧裁后 base 仍在左上角，因此 generator 还保证 dock 后 plug 不会伸到
#   base 左边；做法是两块都以各自 bbox 左边为 seam 的参考原点，并在 solver 中按 bbox 左边对齐。
#
# e) 视觉可观察性
# - 解题者看到的核心差异应是：输入中没有可用的特殊 connector 颜色，输出拼接后的接缝
#   来自“上块底边”和“下块顶边”形状完全对应。
# - seam pattern 总是带 gap，这样肉眼能看到“列模式匹配”，而不是把它误读成连续条长匹配。
#
# f) 可视化整洁度
# - 两个 sprite 都是单色细长连通形，背景干净；输入阶段仅有两块物体，不再叠加额外标记。
# - 输出里两块只在相邻两行形成接缝，不共享像素，因此轮廓关系清晰。
#
# g) 颜色变化可推断性
# - 颜色不再承担匹配作用，固定只用 `bg=1` 和 `obj=4`。
# - 这避免了解题者把注意力误放到 connector 颜色，也避免“颜色映射”成为新维度。


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_cols(pts: set[tuple[int, int]], which: str) -> tuple[int, ...]:
    r0, c0, r1, _c1 = _bbox_of_pts(pts)
    row = r0 if which == "top" else r1
    cols = sorted(int(c - c0) for r, c in pts if int(r) == row)
    assert cols
    return tuple(cols)


def _crop(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _has_gap(cols: tuple[int, ...]) -> bool:
    return any(int(b - a) > 1 for a, b in zip(cols, cols[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites by matching edge profiles, then tight-crop.

    Input structure:
    - Background color `bg=1`, object color `obj=4`.
    - Exactly two 8-connected foreground sprites.
    - The base sprite is anchored at the grid origin.

    Rule:
    - Identify the base sprite (bbox top-left closest to `(0,0)`).
    - Compare the base sprite's bottom-edge foreground columns with the plug sprite's
      top-edge foreground columns, both normalized by each sprite's bbox left edge.
    - Translate the plug so it sits exactly one row below the base and their edge
      column patterns align.
    - Output the tight crop around the merged shape.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    r0a, c0a, r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
        br0, bc0, br1 = r0a, c0a, r1a
        pr0, pc0 = r0b, c0b
    else:
        base, plug = B, A
        br0, bc0, br1 = r0b, c0b, r1b
        pr0, pc0 = r0a, c0a

    base_bottom = _edge_cols(base, "bottom")
    plug_top = _edge_cols(plug, "top")
    assert base_bottom == plug_top

    tr = int(br1 + 1 - pr0)
    tc = int(bc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    assert base.isdisjoint(plug_shift)

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    assert rr0 == br0 and cc0 == bc0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])
    for r, c in plug_shift:
        assert int(out[int(r - rr0), int(c - cc0)]) == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(g[int(r - tr), int(c - tc)])

    out[out == int(ctrl)] = int(obj)
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (output-first: build docked state via matched edge profile)
# ---------------------------------------------------------------------------


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], forbidden: set[tuple[int, int]] | None = None) -> None:
    forbidden = forbidden or set()
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc] or (nr, nc) in forbidden:
                continue
            nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _choose_seam_cols(width: int) -> tuple[int, ...] | None:
    max_k = min(4, int(width) - 1)
    if max_k < 2:
        return None
    for _ in range(50):
        k = _rng.randint(2, max_k)
        cols = tuple(sorted(_rng.sample(range(int(width)), k)))
        if len(cols) < int(width) and _has_gap(cols):
            return cols
    return None


def _carve_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int], inward_step: int) -> None:
    h, _w = mask.shape
    r, c = map(int, start)
    gr, gc = map(int, goal)
    mask[r, c] = True
    if r == 0 and inward_step > 0:
        r = 1
        mask[r, c] = True
    elif r == h - 1 and inward_step < 0:
        r = h - 2
        mask[r, c] = True
    while (r, c) != (gr, gc):
        moves = []
        if r != gr:
            moves.append((int(r + (1 if gr > r else -1)), c))
        if c != gc:
            moves.append((r, int(c + (1 if gc > c else -1))))
        if not moves:
            break
        if len(moves) == 2:
            nr, nc = moves[0] if _rng.random() < 0.6 else moves[1]
        else:
            nr, nc = moves[0]
        r, c = nr, nc
        mask[r, c] = True


def _make_base_mask(height: int, width: int, seam_cols: tuple[int, ...]) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, 0] = True
    keep = {(0, 0)}
    for c in seam_cols:
        goal_choices = [(int(r), int(cc)) for r, cc in zip(*np.where(mask)) if int(r) < int(height) - 1]
        if not goal_choices:
            return None
        goal = _rng.choice(goal_choices)
        _carve_path(mask, (int(height) - 1, int(c)), goal, inward_step=-1)
        keep.add((int(height) - 1, int(c)))
    lower = max(5, int(mask.sum()))
    upper = min(24, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    target = _rng.randint(lower, upper)
    forbidden = {(int(height) - 1, c) for c in range(int(width)) if c not in seam_cols}
    _grow_shape(mask, target, list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    bottom_cols = tuple(sorted(int(c) for c in np.where(mask[-1])[0]))
    if bottom_cols != seam_cols:
        return None
    if not bool(mask[0, 0]):
        return None
    return mask


def _make_plug_mask(height: int, width: int, seam_cols: tuple[int, ...]) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    anchor_col = int(_rng.choice(seam_cols))
    mask[int(height) - 1, anchor_col] = True
    keep = {(0, int(c)) for c in seam_cols}
    for c in seam_cols:
        goal_choices = [(int(r), int(cc)) for r, cc in zip(*np.where(mask)) if int(r) > 0]
        if not goal_choices:
            return None
        goal = _rng.choice(goal_choices)
        _carve_path(mask, (0, int(c)), goal, inward_step=1)
    lower = max(5, int(mask.sum()))
    upper = min(24, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    target = _rng.randint(lower, upper)
    forbidden = {(0, c) for c in range(int(width)) if c not in seam_cols}
    _grow_shape(mask, target, list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    top_cols = tuple(sorted(int(c) for c in np.where(mask[0])[0]))
    if top_cols != seam_cols:
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    bg, obj = 1, 4

    width = _rng.randint(4, 9)
    seam_cols = _choose_seam_cols(width)
    if seam_cols is None:
        return None

    base_h = _rng.randint(3, 5)
    plug_h_max = min(5, 8 - int(base_h))
    if plug_h_max < 3:
        return None
    plug_h = _rng.randint(3, plug_h_max)

    base_mask = _make_base_mask(base_h, width, seam_cols)
    plug_mask = _make_plug_mask(plug_h, width, seam_cols)
    if base_mask is None or plug_mask is None:
        return None

    base_grid = np.full(base_mask.shape, int(bg), dtype=int)
    base_grid[base_mask] = int(obj)
    plug_grid = np.full(plug_mask.shape, int(bg), dtype=int)
    plug_grid[plug_mask] = int(obj)

    base_sprite = _crop(base_grid, bg)
    plug_sprite = _crop(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    if int(np.sum(base_sprite != bg)) < 5 or int(np.sum(plug_sprite != bg)) < 5:
        return None

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != bg)}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != bg)}
    base_sig = _edge_cols(base_pts, "bottom")
    plug_sig = _edge_cols(plug_pts, "top")
    if base_sig != plug_sig or not _has_gap(base_sig):
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(5, max(bw, pw) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
