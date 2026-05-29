# Variant of 78332cb0
# Dimension: D9 — 链的连接语义 (match_cond)
# Change: 每对相邻块的连接边缘恰好有1个前景像素在相同偏移处重合... → 边缘完全互补连接
#
# Design analysis
# ===============
# spec_like:
#   unchanged_structure:
#     - 背景色仍为 7，分隔线仍为 6，块数量仍为 3 或 4，块大小仍为 5x5。
#     - 输入仍是被 sep 整行/整列分开的若干 5x5 单色连通精灵；输出仍是按链顺序排成的 1xN 条带。
#     - 仍要求只有一个方向（水平或垂直）存在完整链，另一个方向不存在。
#   D9_core_change:
#     - 原规则：若 A 的 trailing edge 与 B 的 leading edge 在同一偏移处有前景像素重合，则 A -> B 可连接。
#     - 新规则：只看内部三个偏移 {1,2,3}。若 A.trailing 与 B.leading 在这三个偏移上恰好互补，
#       即并集等于 {1,2,3} 且交集为空，则 A -> B 可连接。
#     - 具体例子：
#       - A.right={1,3}, B.left={2}：原题中不连接（无重合），本变体中连接（完全互补）。
#       - A.right={2}, B.left={2}：原题中连接，本变体中不连接（非互补，且有重合）。
#   behavior_trace:
#     - 输入构造阶段：原生成器给每个连接边只指定一个偏移；现在改为给每个连接边指定一个非空真子集，
#       后继块的对应边必须取其补集，因此相邻块边缘会出现明显“你有我无 / 你无我有”的锯齿咬合。
#     - solver 阶段：分块、识别分隔线、渲染输出都不变；只把 `_find_chain().ok()` 从“交集非空”
#       改成“交集为空且并集覆盖全部内部偏移”。
#   complexity_calibration:
#     - 连接 profile 只允许 {1,2,3} 的非空真子集，共 6 种，复杂度和原题同量级，但视觉差异明显。
#     - 不允许使用空集或全集作为内部连接边，避免“整边全空/全满”让锯齿感消失，也减少退化到弱信号样例。
#     - 仍保留唯一链筛选，避免 profile 重复造成多种排列都成立。
#   implicit_assumptions_review:
#     - 原 `main._contacts()` 仅把边缘前景编码成偏移集合，这在变体中仍成立，不需要改。
#     - 原 `main._find_chain().ok()` 依赖“共享偏移即可连接”；这在变体中失效，必须改成互补判定。
#     - 原 generator 的 `_base_mask(left_offset, right_offset)` 假设每边最多一个接触点；这在变体中失效，
#       必须改成接受一个偏移集合，并验证边缘集合精确匹配。
#     - 原 generator 的 `_count_orders()` 也基于交集判定，必须同步替换；否则唯一性检查会和 solver 不一致。
#     - `main` 新增的关键假设是：合法连接只发生在内部偏移 {1,2,3} 上；generator 通过构造保证 top/bottom
#       或 left/right 的端点偏移 0/4 不会出现在连接边上。
#   placement_legality:
#     - 仍使用 sep 的整行/整列把块严格隔开，所以块间不重叠、不越界，solver 的 split 逻辑不需改。
#     - 每个块继续保持单连通、单色，并避免出现整行/整列全前景，防止与 sep 检测混淆。
#   visual_observability:
#     - 人类可直接看到：原题是“对位重合”，本变体是“错位互补”。相邻边不会在同一行/列同时亮起，
#       而会把三个内部偏移刚好填满，因此 2-3 组样例足以看出规则变化。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_OFFSETS = (1, 2, 3)
EDGE_UNIVERSE = frozenset(EDGE_OFFSETS)
CONNECT_PROFILES = tuple(
    frozenset(profile)
    for k in (1, 2)
    for profile in itertools.combinations(EDGE_OFFSETS, k)
)
FILLER_CELLS = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400
PROFILE_RETRIES = 80

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _complement(profile: frozenset[int]) -> frozenset[int]:
    return frozenset(EDGE_UNIVERSE.difference(profile))


def _is_complementary(a: set[int] | frozenset[int], b: set[int] | frozenset[int]) -> bool:
    aa = frozenset(int(x) for x in a)
    bb = frozenset(int(x) for x in b)
    return aa.isdisjoint(bb) and aa | bb == EDGE_UNIVERSE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by complementary edge profiles.

    Two consecutive blocks connect iff their facing edges partition the three
    interior offsets {1,2,3}: every interior offset appears on exactly one side.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
                cand.append(int(a[int(r), 0]))
        for c in range(int(a.shape[1])):
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
                segs.append(a[int(start) : int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start) : int(c)].copy())
            start = int(c + 1)
        return segs

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = int(len(blocks))
        if int(axis) == 0:
            out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[int(rr) : int(rr + BLOCK_SIZE), :] = block
                rr += BLOCK_SIZE
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc) : int(cc + BLOCK_SIZE)] = block
            cc += BLOCK_SIZE
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
                g[:r, int(c + 1) :],
                g[int(r + 1) :, :c],
                g[int(r + 1) :, int(c + 1) :],
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
                return _is_complementary(contacts[int(a)]["right"], contacts[int(b)]["left"])
            return _is_complementary(contacts[int(a)]["bot"], contacts[int(b)]["top"])

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    blocks = _split_blocks()
    assert blocks, "must extract at least one block"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        return _render_strip([blocks[i] for i in chain_h], axis=1)
    if chain_v is not None and chain_h is None:
        return _render_strip([blocks[i] for i in chain_v], axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


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
        return _is_complementary(contacts[a][lead], contacts[b][trail])

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


def _add_side(mask: np.ndarray, rows: frozenset[int], side: str) -> None:
    for row in rows:
        r = int(row)
        if side == "left":
            mask[r, 0:3] = True
        else:
            mask[r, 2:BLOCK_SIZE] = True
        lo = min(r, 2)
        hi = max(r, 2)
        mask[lo : hi + 1, 2] = True


def _base_mask(left_rows: frozenset[int], right_rows: frozenset[int]) -> np.ndarray | None:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    _add_side(mask, left_rows, "left")
    _add_side(mask, right_rows, "right")
    if not mask.any():
        return None
    if mask.sum() > FG_AREA_MAX:
        return None
    while mask.sum() < FG_AREA_MIN:
        fillers = [(r, c) for r, c in FILLER_CELLS if not mask[r, c]]
        if not fillers:
            break
        r, c = random.choice(fillers)
        mask[r, c] = True
    target_area = random.randint(max(FG_AREA_MIN, int(mask.sum())), FG_AREA_MAX)
    while mask.sum() < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if set(np.where(mask[:, 0])[0].tolist()) != set(left_rows):
        return None
    if set(np.where(mask[:, BLOCK_SIZE - 1])[0].tolist()) != set(right_rows):
        return None
    if mask[0].any() or mask[BLOCK_SIZE - 1].any():
        return None
    if any(mask[r].all() for r in range(BLOCK_SIZE)):
        return None
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(
    color: int,
    *,
    left: frozenset[int] | None = None,
    right: frozenset[int] | None = None,
    top: frozenset[int] | None = None,
    bottom: frozenset[int] | None = None,
) -> np.ndarray | None:
    left_rows = frozenset() if left is None else frozenset(int(x) for x in left)
    right_rows = frozenset() if right is None else frozenset(int(x) for x in right)
    top_cols = frozenset() if top is None else frozenset(int(x) for x in top)
    bottom_cols = frozenset() if bottom is None else frozenset(int(x) for x in bottom)

    if (top is not None) or (bottom is not None):
        mask = _base_mask(bottom_cols, top_cols)
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(left_rows, right_rows)
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = SEP
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), BG, dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + BLOCK_SIZE, :] = SEP
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        anchors = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, anchors):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])

    for _ in range(PROFILE_RETRIES):
        connectors = [random.choice(CONNECT_PROFILES) for _ in range(n - 1)]
        lead_kw, trail_kw = _ORIENT_KW[orient]
        blocks: list[np.ndarray] = []
        for i in range(n):
            kw: dict[str, frozenset[int]] = {}
            if i > 0:
                kw[lead_kw] = _complement(connectors[i - 1])
            if i < n - 1:
                kw[trail_kw] = connectors[i]
            block = _make_block(random.choice(PALETTE), **kw)
            if block is None:
                break
            blocks.append(block)
        if len(blocks) != n:
            continue
        if _count_orders(blocks, orient) != 1:
            continue
        if _count_orders(blocks, _OTHER[orient]) != 0:
            continue
        out = _render_strip(blocks, orient)
        inp = _render_input(blocks, orient)
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
