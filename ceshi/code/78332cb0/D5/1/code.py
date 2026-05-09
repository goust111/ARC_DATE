# Variant of 78332cb0
# Dimension: D5 — 变换操作 (transform)
# Change: 仅平移（将块从打散位置移到链序位置），不旋转不变色 → 匹配边缘擦除
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 输入仍由 3-4 个 5x5 单色连通块组成，块之间由整行/整列分隔色 6 隔开；块内背景仍是 7。
# - 原题输出: 只把这些块按唯一可行的边缘接触顺序排成 1xN 条带，块图案本身不变。
# - 本变体输出: 先按原题方式排成条带，再检查每一对相邻块在链方向上的“匹配边缘像素”。
#   若 A.right 与 B.left 在同一行 r 都有前景，则输出中把 A[r,4] 和 B[r,0] 都擦成背景 7。
#   竖直链同理，擦除 A[4,c] 与 B[0,c]。
# - 例子: 横向链中左块右边缘在第 2 行有色点、右块左边缘第 2 行也有色点。原题会保留这两个点；
#   变体会把这两个点都改成背景，只留下块外露边缘的前景。排序依据不变，但最终图案出现内部“缺口”。
#
# b) 复杂度校准
# - 不增加旋转、翻转、变色，避免自由度过多。
# - 擦除只发生在相邻匹配边缘，且每对相邻块的共享偏移仍限制在 {1,2,3} 内，视觉上稳定且容易比较。
# - 仍保留“唯一方向 + 唯一排列”的原始约束，避免排序歧义。
#
# c) 隐式假设审查
# - 原 main() 的 `_find_chain` 依赖输入块的边缘接触像素存在；因此输入不能预先擦除，只能在输出渲染时擦除。
# - 原 `_render_strip` 默认直接复制块；在变体中这会错误保留内部接触像素，因此必须改成“渲染前复制块并定点擦除”。
# - 原 `_contacts` 只读边缘前景集合，这一前提仍成立，因为输入块结构未变。
# - 原 `_split_blocks` 假设布局只包含 2x2 或 1xN 条带，这一前提仍由 generator 保证，不需要扩展。
#
# d) 放置合法性
# - 输入布局继续使用原题的 2x2/条带布局，块大小和分隔线厚度不变，因此不会重叠或越界。
# - 输出布局仍在块之间保留 1 格 sep 分隔线；擦除只改块边界像素，不会覆盖分隔线，也不会改变网格尺寸。
#
# e) 视觉可观察性
# - 人类能从 2-4 个样例中看到: 排序规则与原题一致，但所有内部连接位置都被“挖空”成背景。
# - 因为每个样例至少有 2 个或 3 个相邻关系，输出中会稳定出现 4 或 6 个被擦除的边缘像素，差异明显。
#
# f) 可视化整洁度
# - 不引入额外标记、不叠放对象，仍是整洁的 5x5 条带，只是条带内部边界更“空”。
# - 分隔色 6 保持不变，观察者能清楚分辨哪些像素属于块本身、哪些是块间分隔。
#
# g) 颜色变化可推断性
# - 不涉及颜色映射；擦除统一改成背景 7。规则由位置关系触发，而不是由颜色值触发。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by matching edge foreground pixels,
    then erase the matched boundary pixels on both sides of each adjacency.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        H, W = a.shape
        cand: list[int] = []
        for r in range(H):
            if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
                cand.append(int(a[int(r), 0]))
        for c in range(W):
            if bool((a[:, int(c)] == int(a[0, int(c)])).all()) and int(a[0, int(c)]) != int(bg):
                cand.append(int(a[0, int(c)]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[int(r), :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, int(c)] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if int(r) > int(start):
                segs.append(a[int(start): int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start): int(c)].copy())
            start = int(c + 1)
        return segs

    def _split_blocks() -> list[np.ndarray]:
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            r = int(rows[0])
            c = int(cols[0])
            return [
                g[:r, :c],
                g[:r, int(c + 1):],
                g[int(r + 1):, :c],
                g[int(r + 1):, int(c + 1):],
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _contacts(block: np.ndarray) -> dict[str, set[int]]:
        bbg = int(_most_common(block))
        m = block != int(bbg)
        return {
            "top": set(np.where(m[0, :])[0].tolist()),
            "bot": set(np.where(m[-1, :])[0].tolist()),
            "left": set(np.where(m[:, 0])[0].tolist()),
            "right": set(np.where(m[:, -1])[0].tolist()),
        }

    def _find_chain(blocks: list[np.ndarray], orient: str):
        cont = [_contacts(b) for b in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return len(cont[int(a)]["right"] & cont[int(b)]["left"]) > 0
            return len(cont[int(a)]["bot"] & cont[int(b)]["top"]) > 0

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    def _erase_shared_edges(blocks: list[np.ndarray], orient: str) -> list[np.ndarray]:
        erased = [block.copy() for block in blocks]
        cont = [_contacts(block) for block in blocks]
        for i in range(len(erased) - 1):
            if orient == "h":
                shared = cont[i]["right"] & cont[i + 1]["left"]
                for r in shared:
                    erased[i][int(r), -1] = int(bg)
                    erased[i + 1][int(r), 0] = int(bg)
            else:
                shared = cont[i]["bot"] & cont[i + 1]["top"]
                for c in shared:
                    erased[i][-1, int(c)] = int(bg)
                    erased[i + 1][0, int(c)] = int(bg)
        return erased

    def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
        blocks = _erase_shared_edges(blocks, orient)
        n = int(len(blocks))
        if orient == "v":
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[int(rr): int(rr + 5), :] = block
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc): int(cc + 5)] = block
            cc += 5
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

    blocks = _split_blocks()
    assert not (not blocks)

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(ordered, orient="h")
    if chain_v is not None and chain_h is None:
        ordered = [blocks[i] for i in chain_v]
        return _render_strip(ordered, orient="v")
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"


# --------------- generator ---------------

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != BG
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _interior_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    interior = mask[1:4, 1:4]
    has_neighbor = np.zeros_like(interior)
    has_neighbor[1:, :] |= interior[:-1, :]
    has_neighbor[:-1, :] |= interior[1:, :]
    has_neighbor[:, 1:] |= interior[:, :-1]
    has_neighbor[:, :-1] |= interior[:, 1:]
    has_neighbor[0, :] |= mask[0, 1:4]
    has_neighbor[2, :] |= mask[4, 1:4]
    has_neighbor[:, 0] |= mask[1:4, 0]
    has_neighbor[:, 2] |= mask[1:4, 4]
    candidates = has_neighbor & ~interior
    rs, cs = np.where(candidates)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int) -> np.ndarray | None:
    S = BLOCK_SIZE
    mask = np.zeros((S, S), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2):max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:S] = True
        mask[min(right_offset, 2):max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while mask.sum() < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if mask[:, S - 1].any():
            return None
    elif np.where(mask[:, S - 1])[0].tolist() != [right_offset]:
        return None

    if mask[0].any() or mask[S - 1].any():
        return None
    if any(mask[r].all() for r in range(S)):
        return None
    if any(mask[:, c].all() for c in range(S)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(
    color: int,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    if top is not None or bottom is not None:
        mask = _base_mask(bottom, top, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(left, right, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = color
    return block


def _erase_shared_edges(blocks: list[np.ndarray], orient: str) -> list[np.ndarray]:
    erased = [block.copy() for block in blocks]
    contacts = [_edge_sets(block) for block in blocks]
    for i in range(len(erased) - 1):
        lead, trail = _EDGE_PAIRS[orient]
        shared = contacts[i][lead] & contacts[i + 1][trail]
        if orient == "h":
            for r in shared:
                erased[i][int(r), -1] = BG
                erased[i + 1][int(r), 0] = BG
        else:
            for c in shared:
                erased[i][-1, int(c)] = BG
                erased[i + 1][0, int(c)] = BG
    return erased


def _render_strip(blocks: list[np.ndarray], orient: str, *, erase_shared: bool) -> np.ndarray:
    tiles = _erase_shared_edges(blocks, orient) if erase_shared else [block.copy() for block in blocks]
    n = len(tiles)
    S = BLOCK_SIZE
    stride = S + 1
    if orient == "h":
        out = np.full((S, n * S + (n - 1)), BG, dtype=int)
        for i, block in enumerate(tiles):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + S] = SEP
    else:
        out = np.full((n * S + (n - 1), S), BG, dtype=int)
        for i, block in enumerate(tiles):
            paste(out, block, i * stride, 0)
            if i < n - 1:
                out[i * stride + S, :] = SEP
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        S = BLOCK_SIZE
        grid = np.full((2 * S + 1, 2 * S + 1), BG, dtype=int)
        grid[S, :] = SEP
        grid[:, S] = SEP
        for block, (r0, c0) in zip(scrambled, [(0, 0), (0, S + 1), (S + 1, 0), (S + 1, S + 1)]):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient], erase_shared=False)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    out = _render_strip(blocks, orient, erase_shared=True)
    inp = _render_input(blocks, orient)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
