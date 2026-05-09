# Variant of 78332cb0
# Dimension: D5 — 变换操作 (transform)
# Change: 仅平移（将块从打散位置移到链序位置），不旋转不变色... → 链序旋转
#
# Design analysis for this D5 variant
# -----------------------------------
# spec_like:
#   preserved_structure:
#     background: 7
#     separator: 6
#     blocks:
#       count: 3-4
#       size: 5x5
#       input_state: 打散摆放, 姿态不变
#       matching_rule: 仍然只靠边缘前景像素接触偏移恢复唯一链顺序与链方向
#   changed_output_rule:
#     chain_rotation:
#       position_indexing: 0-based along recovered chain
#       transform: 第 k 个块顺时针旋转 90*k 度
#       layout: 旋转后再按原链方向拼成 1xN 条带, 仍用 sep=6 分隔
#
# a) 具体化与行为追踪
#   小例子: 若恢复出的水平链顺序为 [A,B,C,D], 原题输出是 [A,B,C,D] 原样横排。
#   现在输出改为 [rot0(A), rot1(B), rot2(C), rot3(D)] 横排。
#   输入构造不需要提前旋转块; 旋转只发生在“solver 已经确定顺序之后”的最后一步。
#   因此受影响的环节只有 output rendering, 不是 block extraction 或 chain finding。
#
# b) 复杂度校准
#   若继续允许任意随机单色小块, 很容易出现 180/90 度对称, 使旋转不可见, 样例退化成原题。
#   所以 generator 额外拒绝所有与自身任一非平凡旋转相同的块。
#   这样仍保留原题的单色连通精灵设定, 只做 D5 所需的最小适配。
#
# c) 对原 main() 的隐式假设审查
#   - _split_blocks(): 只假设输入块是 5x5 并由完整分隔线切开。变体不改输入, 仍成立。
#   - _contacts(): 只从输入块读取接触边。若在求链前旋转块会破坏该假设; 因此 main 必须先求链, 后旋转。
#   - _find_chain(): 依赖输入中的原始接触关系唯一。变体不改匹配规则, 仍成立。
#   - _render_strip(): 原版直接粘贴 ordered blocks。这里若继续无条件粘贴原块, 会退化回原题。
#     必须先对 ordered[i] 施加 rot90(-i) 后再渲染。
#
# d) 放置合法性
#   所有块仍是 5x5 正方形, 旋转后尺寸不变, 所以条带尺寸公式、分隔线位置、2x2 输入布局都无需修改。
#   input 继续使用未旋转块, 避免输入块互相覆盖或破坏原有唯一链判定。
#
# e) 视觉可观察性
#   人类能从 2-4 个样例看出: 同一类任务里, 排序后的第 1/2/3 个块分别多转 90/180/270 度。
#   为了让这个规律稳定可见, generator 要求:
#   - 每个块与其 90/180/270 度旋转均不同
#   - 样本中至少有一个非零旋转位置真正改变像素排布; 实际上这里强制所有块都满足
#
# f) 可视化整洁度
#   不引入额外标记或覆盖层, 仍保持“若干 5x5 块 + 单像素 sep 线”的干净布局。
#   旋转只在块内部重排像素, 不会叠加新元素。
#
# g) 颜色变化可推断性
#   不涉及颜色映射。颜色仍只用于区分不同单色块, 推理核心是位置决定的旋转量。
#
# h) 量化反退化指标
#   一个样本可计算 changed_positions = #{k>0 | rot_k(block_k) != block_k}。
#   本 generator 通过“每个块都非旋转对称”保证 changed_positions = n-1, 因而不会退化为原题。
#
# i) main() 的新增假设与 generator 保证
#   新假设: 至少在非零旋转位置上, 旋转应该对视觉有意义。
#   保证方式: generator 拒绝所有旋转对称块, 并继续用 main(input)==expected_rotated_output 做闭环验证。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by matching edge foreground pixels,
    then rotate each block by its 0-based position in the recovered chain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        H, W = a.shape
        cand: list[int] = []
        for r in range(H):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
                cand.append(int(a[r, 0]))
        for c in range(W):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(a.shape[0]) if bool((a[r, :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(a.shape[1]) if bool((a[:, c] == int(sep)).all())]

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

    def _rotate_chain_block(block: np.ndarray, pos: int) -> np.ndarray:
        return np.rot90(block, k=(-(int(pos) % 4)))

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = int(len(blocks))
        if int(axis) == 0:
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[int(rr): int(rr + 5), :] = _rotate_chain_block(block, i)
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc): int(cc + 5)] = _rotate_chain_block(block, i)
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
        m = block != int(bbg)
        return {
            "top": set(np.where(m[0, :])[0].tolist()),
            "bot": set(np.where(m[-1, :])[0].tolist()),
            "left": set(np.where(m[:, 0])[0].tolist()),
            "right": set(np.where(m[:, -1])[0].tolist()),
        }

    def _find_chain(blocks: list[np.ndarray], orient: str):
        cont = [_contacts(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return bool(cont[int(a)]["right"] & cont[int(b)]["left"])
            return bool(cont[int(a)]["bot"] & cont[int(b)]["top"])

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    blocks = _split_blocks()
    assert blocks

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(ordered, axis=1)
    if chain_v is not None and chain_h is None:
        ordered = [blocks[i] for i in chain_v]
        return _render_strip(ordered, axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


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

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _rotate_cw(block: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(block, k=(-(int(quarter_turns) % 4)))


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
        mask[min(left_offset, 2): max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:S] = True
        mask[min(right_offset, 2): max(right_offset, 2) + 1, 2] = True

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


def _is_rotation_distinct(block: np.ndarray) -> bool:
    return all(not np.array_equal(block, _rotate_cw(block, turns)) for turns in (1, 2, 3))


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
    if not _is_rotation_distinct(block):
        return None
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

    return _render_strip(scrambled, _OTHER[output_orient])


def _rotated_chain(blocks: list[np.ndarray]) -> list[np.ndarray]:
    return [_rotate_cw(block, i) for i, block in enumerate(blocks)]


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

    rotated_blocks = _rotated_chain(blocks)
    if sum(int(not np.array_equal(a, b)) for a, b in zip(blocks, rotated_blocks)) != n - 1:
        return None

    out = _render_strip(rotated_blocks, orient)
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
