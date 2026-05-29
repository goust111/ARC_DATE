# Variant of 78332cb0
# Dimension: D13 — 多步处理 (multi_step)
# Change: 单步处理：找到唯一有效链并输出条带... → 链接后着色

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 输入仍然是被分隔线打散的 3-4 个 5x5 单色精灵块；连接规则完全沿用原题：
#   相邻块在链方向上的对应边缘必须共享同一个前景接触偏移。
# - 第一步仍是原题：先把块按唯一可行方向排成 1xN 条带。
# - 第二步新增：排好链后，不直接输出原块，而是按链方向对每个块的前景重新着色。
#   * 水平链：前景中 `col<2` 的像素取前一个块的原始颜色（若不存在前邻居则保留本块原色）；
#     `col==2` 保留本块原色；`col>2` 取后一个块的原始颜色（若不存在后邻居则保留本块原色）。
#   * 垂直链：同理改成按 `row<2 / row==2 / row>2` 取 前邻居 / 自身 / 后邻居 颜色。
# - 小例子：若水平链颜色依次是 A(2), B(4), C(8)，原题输出还是 [2][4][8] 三个单色块；
#   变体输出则是：
#   * 第 1 块：左半仍 2，中心 2，右半变 4
#   * 第 2 块：左半变 2，中心 4，右半变 8
#   * 第 3 块：左半变 4，中心 8，右半仍 8
#   差异在于：排序仍依赖边缘匹配，但最终颜色必须等排序完成后才能确定。
#
# b) 复杂度校准
# - 若只做“整块复制某个邻居颜色”，第二步会过于弱。这里让内块同时显式展示前邻居色、自身色、后邻居色，
#   形成 2-3 色块，能肉眼看出确实经历了第二步。
# - 为避免颜色关系不明显，generator 强制每条链上的块使用互不相同的原始颜色。
#   这样输出中的颜色流向稳定可见，且不需要解题者猜隐式色表。
#
# c) 隐式假设审查（相对原 main 的相关段）
# - 原 `_contacts()` 依赖“输入块仍是单色且背景占多数”，本变体保持输入不变，因此这一前提仍成立。
# - 原 `_find_chain()` 只看边缘接触集合；输出变成多色不会影响它，因为链判定发生在输入块上。
# - 原来 `_render_strip(ordered)` 直接输出排序结果；这里必须先 `ordered -> recolored -> render`，
#   否则会退化回原任务。
# - 原题里块内只有一种前景色，所以“块颜色”可以简单取唯一非背景色；本变体输出块可能多色，
#   因此求块原色时只能对输入块做，不能对已经着色后的输出块再复用该逻辑。
#
# d) 放置合法性
# - 输入放置方式沿用原题：3 块时一定是条带，4 块时可为 2x2 或条带；块之间由 1 像素 sep 全分隔，
#   因此不会重叠，也不会越界。
# - 输出仍是 1xN 条带，尺寸公式不变；第二步只改块内前景颜色，不改任何占位像素，因此版式合法。
#
# e) 视觉可观察性
# - 从样例上，人类先能看出原题的“拼链”，再能看出输出里每个块沿链方向吸收了相邻块的颜色。
# - 颜色变化只依赖可见的链邻接关系，不依赖颜色编号大小、隐藏计数或不可见元数据。
#
# f) 可视化整洁度
# - 只在块内部着色，不增加额外标记或叠层；分隔线、背景、块布局保持原题整洁风格。
# - 使用 5x5 块自身已有的“左臂/中脊/右臂”或“上臂/中脊/下臂”来承载三段颜色，不会造成遮挡。
#
# g) 颜色变化可推断性
# - 每条链使用互异原色，且内块通常会同时出现三种颜色：前邻居色 / 自身色 / 后邻居色。
# - 颜色来源由位置决定（链方向上的前、中、后），不是任意映射，因此 2-3 个样例足以归纳。
#
# 9) 对新 main() 的额外输入假设
# - 假设输入块仍是原题风格：单连通、单前景色、背景占多数、连接边只有链方向接触像素。
# - 假设每个块在链主轴中心线上至少有前景，generator 通过 `_base_mask()` 的中心脊构造保证这一点，
#   这样第二步中的“中心列/行保留自身颜色”一定可见。
# - 假设存在且仅存在一个合法链方向和顺序；generator 仍用穷举计数强制保证。


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


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.flatten())).most_common(1)[0][0])


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    for r in range(int(a.shape[0])):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(int(a.shape[1])):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if int(r) > int(start):
            segs.append(a[start:r, :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, start:c].copy())
        start = int(c + 1)
    return segs


def _render_strip(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    n = int(len(blocks))
    if int(axis) == 0:
        out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + BLOCK_SIZE, :] = block
            rr += BLOCK_SIZE
            if i < n - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out

    out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, cc : cc + BLOCK_SIZE] = block
        cc += BLOCK_SIZE
        if i < n - 1:
            out[:, cc] = int(sep)
            cc += 1
    return out


def _split_blocks(g: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(g, sep)
    cols = _full_sep_cols(g, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        return [
            g[:r, :c].copy(),
            g[:r, c + 1 :].copy(),
            g[r + 1 :, :c].copy(),
            g[r + 1 :, c + 1 :].copy(),
        ]
    if rows:
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _contacts(block: np.ndarray) -> dict[str, set[int]]:
    bbg = _most_common(block)
    mask = block != int(bbg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _find_chain(blocks: list[np.ndarray], orient: str) -> tuple[int, ...] | None:
    contacts = [_contacts(block) for block in blocks]

    def ok(a: int, b: int) -> bool:
        if orient == "h":
            return bool(contacts[a]["right"] & contacts[b]["left"])
        return bool(contacts[a]["bot"] & contacts[b]["top"])

    idx = list(range(len(blocks)))
    for perm in itertools.permutations(idx):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
            return tuple(int(x) for x in perm)
    return None


def _single_fg_color(block: np.ndarray) -> int:
    bbg = _most_common(block)
    cols = sorted({int(x) for x in block.flatten() if int(x) != int(bbg)})
    assert len(cols) == 1, "input blocks must be single-color before recoloring"
    return int(cols[0])


def _fg_color_count(block: np.ndarray) -> int:
    cols = {int(x) for x in block.flatten() if int(x) != BG}
    return len(cols)


def _recolor_block(
    block: np.ndarray,
    orient: str,
    prev_color: int | None,
    self_color: int,
    next_color: int | None,
) -> np.ndarray:
    bbg = _most_common(block)
    mask = block != int(bbg)
    out = block.copy()
    rs, cs = np.where(mask)
    for r, c in zip(rs.tolist(), cs.tolist()):
        if orient == "h":
            if int(c) < 2:
                out[int(r), int(c)] = int(prev_color if prev_color is not None else self_color)
            elif int(c) > 2:
                out[int(r), int(c)] = int(next_color if next_color is not None else self_color)
            else:
                out[int(r), int(c)] = int(self_color)
        else:
            if int(r) < 2:
                out[int(r), int(c)] = int(prev_color if prev_color is not None else self_color)
            elif int(r) > 2:
                out[int(r), int(c)] = int(next_color if next_color is not None else self_color)
            else:
                out[int(r), int(c)] = int(self_color)
    return out


def _recolor_chain(blocks: list[np.ndarray], orient: str) -> list[np.ndarray]:
    colors = [_single_fg_color(block) for block in blocks]
    recolored: list[np.ndarray] = []
    for i, block in enumerate(blocks):
        prev_color = colors[i - 1] if i > 0 else None
        next_color = colors[i + 1] if i + 1 < len(blocks) else None
        recolored.append(_recolor_block(block, orient, prev_color, colors[i], next_color))
    return recolored


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)

    blocks = _split_blocks(g, sep)
    assert blocks, "expected at least one block"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"

    if chain_h is not None:
        ordered = [blocks[i] for i in chain_h]
        recolored = _recolor_chain(ordered, "h")
        return _render_strip(recolored, axis=1, bg=bg, sep=sep)

    ordered = [blocks[i] for i in chain_v]
    recolored = _recolor_chain(ordered, "v")
    return _render_strip(recolored, axis=0, bg=bg, sep=sep)


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
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
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
        if mask[:, BLOCK_SIZE - 1].any():
            return None
    elif np.where(mask[:, BLOCK_SIZE - 1])[0].tolist() != [right_offset]:
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
    block[mask] = int(color)
    return block


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
        slots = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, axis=1 if _OTHER[output_orient] == "h" else 0, bg=BG, sep=SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    colors = random.sample(PALETTE, n)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(colors[i], **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    recolored = _recolor_chain(blocks, orient)
    if any(_fg_color_count(block) < 2 for block in recolored):
        return None
    if n >= 3 and not any(_fg_color_count(block) >= 3 for block in recolored[1:-1]):
        return None

    plain_out = _render_strip(blocks, axis=1 if orient == "h" else 0, bg=BG, sep=SEP)
    out = _render_strip(recolored, axis=1 if orient == "h" else 0, bg=BG, sep=SEP)
    if np.array_equal(out, plain_out):
        return None

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
