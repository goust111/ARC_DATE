# Variant of 78332cb0
# Dimension: D4 — 目标排列方式 (dst)
# Change: 1xN条带（水平或垂直），块按链序排列，用品红分隔线分开
#   -> 2D网格排列：链顺序固定按 2x2 蛇形展开，3 块时保留一个空格，4 块时填满。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


"""
design_analysis:
  preserved_structure:
    - 输入仍然是若干个 5x5 单色 sprite block，背景色为 7，分隔线为 6。
    - block 的连接规则不变：只有一个方向（横向或纵向）能形成唯一链，依据相邻边缘前景像素 offset 是否重合。
    - generator 仍然采用 output-first：先构造合法链，再打乱生成输入，并用 main 回验。
  changed_dimension:
    dst:
      original: 输出是 1xN 条带，方向与链方向一致。
      variant: 输出统一是 2x2 网格，中心有一条 sep 横线和一条 sep 竖线。
      fill_rule: 链顺序按固定蛇形填充格子 TL -> TR -> BR -> BL。
      truncation: 仅 3 块时只放前三个位置，BL 保持背景空白。
  concrete_behavior:
    - 若链顺序为 A,B,C,D，则输出四格依次是：
        A | B
        -----
        D | C
    - 若链顺序为 A,B,C，则输出为：
        A | B
        -----
        . | C
      其中 "." 是纯背景 5x5 空格，不会被当成 block。
  behavior_trace:
    - 原题流程：从输入拆出 block -> 检测横/纵链 -> 按该方向渲染为 1xN 条带。
    - 变体流程：前两步完全相同；差异只出现在最后一步，改为把已排序链渲染进固定 2x2 蛇形网格。
    - 小例子：若纵向链唯一且顺序为 [red, green, blue]，
      原题输出是 red / green / blue 的竖条；
      变体输出是 top-left=red, top-right=green, bottom-right=blue, bottom-left=空白。
  complexity_calibration:
    - 不引入颜色驱动规则，也不引入多种可选网格尺寸，避免解题者需要猜测隐藏参数。
    - 仍保留 3 块和 4 块两档，保证能看到“二维展开 + 截断空位”这两个可观察现象。
    - 网格填充顺序固定为一种蛇形，避免 2-4 个样例不足以区分 Z 形与蛇形的风险。
  implicit_assumption_audit:
    - 原 main 中 `_render_strip` 假设输出必为 1D；这里必须替换为 `_render_snake_grid`。
    - 原 main 的 `_split_blocks` 在 2x2 输入时默认四个象限都有效；变体为了兼容可能出现的空象限，改成过滤全背景格。
    - 原 main 的 `_find_chain` 返回第一条可行链；在条带任务中 generator 已保证唯一，但变体仍显式枚举并要求“唯一方向 + 唯一排列”，避免二维摆放时沿用错误次序。
    - `_contacts` 依赖 block 背景占多数；generator 继续约束 fg 面积 <= 10，因此该假设仍成立。
  placement_legality:
    - 输出画布固定为 11x11，即 `2*5+1`，四个 5x5 格子由中心十字 sep 分开，不会重叠或越界。
    - 输入沿用原任务合法布局：3 块为条带；4 块为条带或 2x2 网格。block 之间只通过 sep 行/列分隔。
    - 3 块输出中的空格只是背景，不参与输入，也不会影响 solver 的拆块。
  visual_observability:
    - 原题输出总是“只有 sep 行或只有 sep 列”；变体输出每个样例都会同时出现一条完整 sep 行和一条完整 sep 列，二维差异肉眼可见。
    - 3 块样例的空格位置固定在左下角，使“蛇形展开被截断”成为稳定、可学习的结构线索。
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设 block 数量为 3 或 4；generator 固定在这两档。
    - solver 假设恰有一个方向存在唯一合法排列；generator 用 `_count_orders` 显式检查。
    - solver 假设分隔色能由整行/整列常色识别；generator 不在 block 内使用 sep 色。
"""


# -- Configuration constants --------------------------------------------------
BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400
# -----------------------------------------------------------------------------

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}
_SNAKE_POSITIONS = [
    (0, 0),
    (0, BLOCK_SIZE + 1),
    (BLOCK_SIZE + 1, BLOCK_SIZE + 1),
    (BLOCK_SIZE + 1, 0),
]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique edge-contact chain, then place it into a 2x2 snake grid.

    Output layout:
    - 4 blocks: TL -> TR -> BR -> BL
    - 3 blocks: TL -> TR -> BR, with BL left as background
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
                cand.append(int(a[r, 0]))
        for c in range(int(a.shape[1])):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if r > start:
                part = a[start:r, :].copy()
                if np.any(part != int(bg)):
                    segs.append(part)
            start = r + 1
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if c > start:
                part = a[:, start:c].copy()
                if np.any(part != int(bg)):
                    segs.append(part)
            start = c + 1
        return segs

    def _split_blocks() -> list[np.ndarray]:
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            row_bounds = [0] + [r + 1 for r in rows]
            row_ends = rows + [int(g.shape[0])]
            col_bounds = [0] + [c + 1 for c in cols]
            col_ends = cols + [int(g.shape[1])]
            blocks: list[np.ndarray] = []
            for r0, r1 in zip(row_bounds, row_ends):
                for c0, c1 in zip(col_bounds, col_ends):
                    if r1 > r0 and c1 > c0:
                        part = g[r0:r1, c0:c1].copy()
                        if np.any(part != int(bg)):
                            blocks.append(part)
            return blocks
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

    def _find_chains(blocks: list[np.ndarray], orient: str) -> list[tuple[int, ...]]:
        contacts = [_contacts(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return bool(contacts[a]["right"] & contacts[b]["left"])
            return bool(contacts[a]["bot"] & contacts[b]["top"])

        chains: list[tuple[int, ...]] = []
        for perm in itertools.permutations(range(len(blocks))):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
                chains.append(tuple(int(x) for x in perm))
        return chains

    def _render_snake_grid(blocks: list[np.ndarray]) -> np.ndarray:
        n = len(blocks)
        assert 3 <= n <= 4
        S = BLOCK_SIZE
        out = np.full((2 * S + 1, 2 * S + 1), int(bg), dtype=int)
        out[S, :] = int(sep)
        out[:, S] = int(sep)
        for block, (r0, c0) in zip(blocks, _SNAKE_POSITIONS):
            paste(out, block, r0, c0)
        return out

    blocks = _split_blocks()
    assert blocks, "must find at least one block"
    assert len(blocks) in (3, 4), "variant expects 3 or 4 blocks"

    chain_h = _find_chains(blocks, "h")
    chain_v = _find_chains(blocks, "v")

    if len(chain_h) == 1 and not chain_v:
        return _render_snake_grid([blocks[i] for i in chain_h[0]])
    if len(chain_v) == 1 and not chain_h:
        return _render_snake_grid([blocks[i] for i in chain_v[0]])
    assert not chain_h or not chain_v, "ambiguous: both H and V chains found"
    assert len(chain_h) <= 1 and len(chain_v) <= 1, "ambiguous: multiple chains found"
    raise AssertionError("must find exactly one valid chain orientation")


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

    if int(mask.sum()) < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < int(target_area):
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


def _render_output_grid(blocks: list[np.ndarray]) -> np.ndarray:
    S = BLOCK_SIZE
    out = np.full((2 * S + 1, 2 * S + 1), BG, dtype=int)
    out[S, :] = SEP
    out[:, S] = SEP
    for block, (r0, c0) in zip(blocks, _SNAKE_POSITIONS):
        paste(out, block, r0, c0)
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
        row_major_positions = [
            (0, 0),
            (0, S + 1),
            (S + 1, 0),
            (S + 1, S + 1),
        ]
        for block, (r0, c0) in zip(scrambled, row_major_positions):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient])


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

    if _count_orders(blocks, orient) != 1:
        return None
    if _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    out = _render_output_grid(blocks)
    inp = _render_input(blocks, orient)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
