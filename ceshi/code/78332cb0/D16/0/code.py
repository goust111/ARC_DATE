from __future__ import annotations

# Variant of 78332cb0
# Dimension: D16 — 链序输出方向 (grid_shape)
# Change: 输出方向由有效链的方向决定（水平链输出水平条带，垂直链输出垂直条带）
#   → 输出方向取反；先按原题方式找唯一有效链，再把每个 5x5 块旋转 90 度，
#   并沿与链方向相反的方向输出。
#
# Design analysis
# - source_task: None；本变体只围绕原题 D16 的输出方向翻转，不引入其他机制。
# - concrete behavior:
#   input 仍然是被 sep=6 分开的 5x5 精灵块，链方向仍由边缘前景像素重合决定。
#   原题若找到水平链 A->B->C，就输出水平条带 [A,B,C]。
#   本变体改为输出竖直条带，并把 A/B/C 各自旋转 90 度后再从上到下放置。
#   若找到垂直链，则反过来输出水平条带，块也旋转 90 度后从左到右放置。
# - behavior trace:
#   原题水平链靠 A.right ∩ B.left、B.right ∩ C.left 判定顺序。
#   本变体仍先用这一步得到 A,B,C；差异只发生在 render 阶段：
#   水平链时把每块顺时针旋转，使原 right/left 接触边转成 bottom/top，
#   然后竖向堆叠；垂直链时把每块逆时针旋转，使原 bottom/top 接触边转成
#   right/left，然后横向排列。这样既“方向取反”，又保留同序相邻块的接触关系。
# - complexity calibration:
#   链判定、块数、颜色、连通性、唯一性约束全部沿用原题，避免把题简化成只看输出形状。
#   旋转方向只分两档，且完全由链方向决定，没有额外自由度。
# - implicit assumption audit for main():
#   `_find_chain` 依赖“输入块仍是未旋转的原始块、连接发生在原链方向边缘”；
#   生成器继续按原规则造输入，所以这个前提仍成立。
#   原题末尾 `return _render_strip(ordered, axis=...)` 假设输出方向=链方向；
#   这里必须显式替换为“先旋转块，再按相反方向 render”，否则会退化回原题。
#   `_split_blocks`、`_find_sep`、`_contacts` 不依赖输出方向，保持不变即可。
# - placement legality:
#   输出仍是 1xN 条带，separator 仍是单行/单列 6；旋转不改变 5x5 尺寸，不会越界。
#   输入布局仍是原题的 2x2 或单条带打散布局，各块之间由完整 sep 行/列隔开，不重叠。
# - visual observability:
#   仅“方向相反”已能从输出网格长宽直接观察到；进一步让块一起旋转 90 度，
#   可在少量样例中看出不是单纯把条带转置，而是每个块也被一起转向。
# - visual cleanliness:
#   不叠加额外标记；仍只使用 bg、sep 和单色精灵，输出结构与原题同样规整。
# - color inference:
#   颜色不参与排序或旋转决策；颜色只是块身份的可见属性，因此不存在隐式颜色编码。
# - generator guarantees:
#   继续 output-first：先构造原链，再把“旋转后、方向取反”的结果作为目标输出；
#   仅接受 `main(inp) == out`、链方向唯一、且变体输出形状不同于原题输出的样本。

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into the unique valid chain, then invert the strip direction.

    D16 variant rule:
    - detect the unique chain orientation exactly as in the source task;
    - if the chain is horizontal, rotate each block 90° clockwise and stack vertically;
    - if the chain is vertical, rotate each block 90° counterclockwise and place horizontally.
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
                segs.append(a[int(start):int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start):int(c)].copy())
            start = int(c + 1)
        return segs

    def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
        n = int(len(blocks))
        if orient == "v":
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[int(rr):int(rr + 5), :] = block
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc):int(cc + 5)] = block
            cc += 5
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

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
        mask = block != int(bbg)
        return {
            "top": set(np.where(mask[0, :])[0].tolist()),
            "bot": set(np.where(mask[-1, :])[0].tolist()),
            "left": set(np.where(mask[:, 0])[0].tolist()),
            "right": set(np.where(mask[:, -1])[0].tolist()),
        }

    def _find_chain(blocks: list[np.ndarray], orient: str):
        contacts = [_contacts(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return len(contacts[int(a)]["right"] & contacts[int(b)]["left"]) > 0
            return len(contacts[int(a)]["bot"] & contacts[int(b)]["top"]) > 0

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    def _rotate_for_output(blocks: list[np.ndarray], chain_orient: str) -> list[np.ndarray]:
        if chain_orient == "h":
            return [np.rot90(block, 3).copy() for block in blocks]
        return [np.rot90(block, 1).copy() for block in blocks]

    blocks = _split_blocks()
    assert blocks, "must extract at least one block"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"

    if chain_h is not None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(_rotate_for_output(ordered, "h"), "v")

    ordered = [blocks[i] for i in chain_v]
    return _render_strip(_rotate_for_output(ordered, "v"), "h")


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

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}
ROT_K = {"h": 3, "v": 1}


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
    lead, trail = EDGE_PAIRS[orient]

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


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    S = BLOCK_SIZE
    stride = S + 1
    if orient == "h":
        out = np.full((S, n * S + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + S] = SEP
        return out

    out = np.full((n * S + (n - 1), S), BG, dtype=int)
    for i, block in enumerate(blocks):
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

    return _render_strip(scrambled, OTHER[output_orient])


def _rotate_output_blocks(blocks: list[np.ndarray], chain_orient: str) -> list[np.ndarray]:
    return [np.rot90(block, ROT_K[chain_orient]).copy() for block in blocks]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = ORIENT_KW[orient]
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

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, OTHER[orient]) != 0:
        return None

    output_orient = OTHER[orient]
    out = _render_strip(_rotate_output_blocks(blocks, orient), output_orient)
    baseline_out = _render_strip(blocks, orient)
    if out.shape == baseline_out.shape:
        return None

    inp = _render_input(blocks, output_orient)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
