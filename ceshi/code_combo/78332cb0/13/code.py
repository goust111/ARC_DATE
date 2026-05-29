# Combo variant of 78332cb0 (index 13)
# combo_spec: D1/1+D17/1
# Name: 图例驱动颜色边缘匹配

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

"""
design_analysis:
  variant_visual:
    - 输入顶部新增 2 行角落图例，左上角放两组 2x2 颜色对，中间隔 1 列背景；整行宽度其余部分保持背景。
      每组左列表示前一块 facing-edge 上的源颜色，右列表示后一块 facing-edge 上可匹配的目标颜色。
    - 图例下方固定有 1 条整行 SEP，再下方是原题的打散 5x5 块布局（条带或 2x2）。
    - 每条可连接边不再只是“某个 offset 上有 1 个前景点”，而是整条 5 格边序列里在两个相邻内部
      offset 上出现前景颜色；样例内所有连接都复用同一组 offset，所以几何占位本身故意做成歧义。
    - 块主体仍是单连通精灵，但链方向边缘可出现图例里的 2 种接边颜色，因此块变成有限多色。

  behavior_trace:
    - 例：图例为 (2->8) 与 (4->1)，所有连接都占用 edge offset {1,2}。
      A.right = [bg,2,4,bg,bg]，B.left = [bg,8,1,bg,bg]，C.left = [bg,1,8,bg,bg]。
    - 原题逻辑只看 facing 边有没有相同 offset 的前景，于是 A->B 与 A->C 都会被当成可接，因为它们都占用 {1,2}。
    - D1 的“整条边颜色序列相等”若直接套用，会要求 [2,4] == [8,1]，反而一个都接不上。
    - 融合后规则是在整条边上逐格应用图例翻译：2 对应 8、4 对应 1，因此 A->B 全序列合法；
      A->C 在相同几何占位下颜色次序反了，所以非法。
    - 差异集中在原 `_contacts()` / `_find_chain()`：原来可把边缘压成布尔 offset 集合；现在必须保留
      完整 edge sequence，并按图例逐格检查颜色对。

  complexity_calibration:
    - 图例只保留 2 组有向颜色对，控制自由度；但每条链会重复使用它们，保证人眼能从少量样例看出规律。
    - 每条连接边固定只有 2 个前景位，位置来自 {1,2} 或 {2,3} 之一，视觉上清楚且仍保留原题“对齐边缘”的骨架。
    - 块数保留 3 或 4，并偏向 4；这样 3 条链接足以形成“同占位但不同颜色次序”的重复证据。

  implicit_assumptions_review:
    - 原 `_split_blocks()` 默认整张图只有块阵列；加入顶部图例后，这个前提失效，所以 solver 必须先找到第一条
      整行 SEP，把图例带与块区分开。
    - 原 `_contacts()` 用二值化记录是否有前景，会抹掉 D1 与 D17 的关键信息；现在要记录每条边的 5 格颜色序列。
    - 原 `ok(a,b)` 只要求 facing-edge 的 offset 集合相交；融合后必须要求：
      1) 两边前景/背景占位完全一致；
      2) 每个非背景格的颜色对都属于图例；
      3) 至少存在 1 个非背景匹配格。
    - 原 `_find_sep()` 仍可用，因为图例本身不产生整列/整行纯色线；新增的唯一顶部整行纯色线正是 legend 下方的 SEP。
    - 新 solver 额外假设图例项是实心 2x2 颜色列、块在非链方向边缘为空；生成器显式保证这两点。

  placement_legality:
    - 生成器继续采用 output-first：先造合法链，再打乱排列成输入。
    - 每个样例统一选择一组相邻 offset 模式 `(1,2)` 或 `(2,3)`；块掩码由这些 offset 向中轴连通扩展，
      保证单连通、不越界，且面积稳定落在 4-10。
    - 端点块只拥有 1 条链向边，内部块拥有 2 条；非链方向两条边保持为空，从而阻断垂直/水平误连。
    - 图例区固定占顶部 2 行，下面独立隔 1 行 SEP，因此与块区不会重叠。

  visual_observability:
    - 解题者能直接看到两层证据：角落图例给出“左色可接右色”的有向翻译表；块边缘上则出现这些颜色并且
      在同一组 offset 上反复出现。
    - 因为样例内所有候选连接几何占位都相同，只看形状无法排顺序；只有结合图例和颜色次序才能锁定链。
    - 所有图例对均为非恒等映射，所以“颜色完全相同才匹配”的解释会在样例中立即失效。

  visual_tidy:
    - 图例与块区垂直分层，中间用整行 SEP 断开，避免对象叠盖。
    - 每个块只会使用 1 个主体色和最多 2 个接边色，颜色种类受控，不会在 5x5 内变得杂乱。
    - 连接位限制在相邻两个内部 offset，边缘证据集中，肉眼容易比较。

  color_inference:
    - 颜色变化不是隐式编码：图例把颜色翻译规则显式画在输入里，颜色推理完全基于可见结构。
    - 仅 2 组颜色对且在同一样例中重复复现，足以让人从 2-3 个训练样例中归纳“按图例逐格翻译后再匹配”。

  generator_guarantees_for_new_main:
    - 顶部一定有且仅有 1 条先于块区的整行 SEP，可作为裁切锚点。
    - 每条链接边都是完整 edge sequence，且恰好含 2 个非背景匹配位；序列颜色来自 legend 的源/目标两列。
    - 融合规则下恰有 1 个方向、1 个排列成立；占位-only 规则会出现多个排列，而 identity-color 规则为 0，
      因而样本不会退化回原题或回纯 D1。
"""

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
LEGEND_HEIGHT = 2
LEGEND_PAIRS = 2

FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_PATTERNS = [(1, 2), (2, 3)]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 600

_OTHER = {"h": "v", "v": "h"}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    h, w = a.shape
    for r in range(h):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != bg:
            cand.append(int(a[r, 0]))
    for c in range(w):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != bg:
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(a.shape[0]) if bool((a[r, :] == sep).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(a.shape[1]) if bool((a[:, c] == sep).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if r > start:
            segs.append(a[start:r, :].copy())
        start = r + 1
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if c > start:
            segs.append(a[:, start:c].copy())
        start = c + 1
    return segs


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), bg, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = sep
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), bg, dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + BLOCK_SIZE, :] = sep
    return out


def _split_blocks(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    cols = _full_sep_cols(a, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        blocks = [
            a[:r, :c].copy(),
            a[:r, c + 1 :].copy(),
            a[r + 1 :, :c].copy(),
            a[r + 1 :, c + 1 :].copy(),
        ]
    elif rows:
        blocks = _split_by_rows(a, sep)
    elif cols:
        blocks = _split_by_cols(a, sep)
    else:
        blocks = []
    assert blocks, "must find at least one block"
    for block in blocks:
        assert block.shape == (BLOCK_SIZE, BLOCK_SIZE), f"unexpected block shape {block.shape}"
    return blocks


def _parse_legend(a: np.ndarray, bg: int) -> set[tuple[int, int]]:
    assert a.shape[0] >= LEGEND_HEIGHT, "legend band must be at least 2 rows high"
    band = a[:LEGEND_HEIGHT, :]
    pairs: set[tuple[int, int]] = set()
    c = 0
    while c < band.shape[1]:
        if bool((band[:, c] == bg).all()):
            c += 1
            continue
        assert c + 1 < band.shape[1], "legend entry must have width 2"
        left = band[:, c]
        right = band[:, c + 1]
        assert bool((left == int(left[0])).all()) and int(left[0]) != bg
        assert bool((right == int(right[0])).all()) and int(right[0]) != bg
        pairs.add((int(left[0]), int(right[0])))
        c += 2
        if c < band.shape[1] and bool((band[:, c] == bg).all()):
            c += 1
    assert len(pairs) == LEGEND_PAIRS, "legend must contain exactly two color pairs"
    return pairs


def _edge_sequences(block: np.ndarray) -> dict[str, tuple[int, ...]]:
    return {
        "top": tuple(int(v) for v in block[0, :]),
        "bot": tuple(int(v) for v in block[BLOCK_SIZE - 1, :]),
        "left": tuple(int(v) for v in block[:, 0]),
        "right": tuple(int(v) for v in block[:, BLOCK_SIZE - 1]),
    }


def _occupancy(seq: tuple[int, ...], bg: int) -> tuple[bool, ...]:
    return tuple(int(v) != bg for v in seq)


def _match_translated(seq_a: tuple[int, ...], seq_b: tuple[int, ...], bg: int, allowed_pairs: set[tuple[int, int]]) -> bool:
    matched_fg = False
    for ca, cb in zip(seq_a, seq_b):
        ca = int(ca)
        cb = int(cb)
        if ca == bg and cb == bg:
            continue
        if ca == bg or cb == bg:
            return False
        if (ca, cb) not in allowed_pairs:
            return False
        matched_fg = True
    return matched_fg


def _count_orders(
    blocks: list[np.ndarray],
    orient: str,
    bg: int,
    *,
    mode: str,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> int:
    contacts = [_edge_sequences(block) for block in blocks]
    lead, trail = ("right", "left") if orient == "h" else ("bot", "top")

    def ok(a: int, b: int) -> bool:
        seq_a = contacts[a][lead]
        seq_b = contacts[b][trail]
        if mode == "translated":
            assert allowed_pairs is not None
            return _match_translated(seq_a, seq_b, bg, allowed_pairs)
        if mode == "occupancy":
            occ_a = _occupancy(seq_a, bg)
            occ_b = _occupancy(seq_b, bg)
            return occ_a == occ_b and any(occ_a)
        if mode == "identity":
            return seq_a == seq_b and any(int(v) != bg for v in seq_a)
        raise ValueError(f"unknown mode {mode}")

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _signature_from_order(offsets: tuple[int, int], colors: tuple[int, int], bg: int) -> tuple[int, ...]:
    seq = [bg] * BLOCK_SIZE
    seq[int(offsets[0])] = int(colors[0])
    seq[int(offsets[1])] = int(colors[1])
    return tuple(seq)


def _map_signature(sig: tuple[int, ...], mapping: dict[int, int], bg: int) -> tuple[int, ...]:
    out = []
    for color in sig:
        color = int(color)
        out.append(bg if color == bg else int(mapping[color]))
    return tuple(out)


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


def _build_mask(left_offsets: tuple[int, ...], right_offsets: tuple[int, ...]) -> np.ndarray | None:
    if not left_offsets and not right_offsets:
        return None
    active = sorted(set(int(v) for v in left_offsets) | set(int(v) for v in right_offsets))
    assert active
    if active[0] < 1 or active[-1] > 3:
        return None
    if active[-1] - active[0] > 1:
        return None

    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for r in left_offsets:
        mask[int(r), 0:3] = True
    for r in right_offsets:
        mask[int(r), 2:BLOCK_SIZE] = True
    mask[int(active[0]) : int(active[-1]) + 1, 2] = True

    base_area = int(mask.sum())
    if not (FG_AREA_MIN <= base_area <= FG_AREA_MAX):
        return None

    target_area = random.randint(base_area, FG_AREA_MAX)
    while int(mask.sum()) < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[int(r), int(c)] = True

    if tuple(int(v) for v in np.where(mask[:, 0])[0].tolist()) != tuple(left_offsets):
        return None
    if tuple(int(v) for v in np.where(mask[:, BLOCK_SIZE - 1])[0].tolist()) != tuple(right_offsets):
        return None
    if mask[0].any() or mask[BLOCK_SIZE - 1].any():
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _assign_edge_colors(block: np.ndarray, edge: str, seq: tuple[int, ...] | None, bg: int) -> None:
    if seq is None:
        return
    if edge in {"left", "right"}:
        c = 0 if edge == "left" else BLOCK_SIZE - 1
        for r, color in enumerate(seq):
            color = int(color)
            if color == bg:
                continue
            assert block[r, c] != bg
            block[r, c] = color
        return
    r = 0 if edge == "top" else BLOCK_SIZE - 1
    for c, color in enumerate(seq):
        color = int(color)
        if color == bg:
            continue
        assert block[r, c] != bg
        block[r, c] = color


def _make_block(
    body_color: int,
    bg: int,
    *,
    left: tuple[int, ...] | None = None,
    right: tuple[int, ...] | None = None,
    top: tuple[int, ...] | None = None,
    bottom: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    if top is not None or bottom is not None:
        bottom_offsets = tuple(int(i) for i, color in enumerate(bottom or ()) if int(color) != bg)
        top_offsets = tuple(int(i) for i, color in enumerate(top or ()) if int(color) != bg)
        mask = _build_mask(bottom_offsets, top_offsets)
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
        block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
        block[mask] = int(body_color)
        _assign_edge_colors(block, "top", top, bg)
        _assign_edge_colors(block, "bottom", bottom, bg)
        if np.any(block[:, 0] != bg) or np.any(block[:, BLOCK_SIZE - 1] != bg):
            return None
        return block

    left_offsets = tuple(int(i) for i, color in enumerate(left or ()) if int(color) != bg)
    right_offsets = tuple(int(i) for i, color in enumerate(right or ()) if int(color) != bg)
    mask = _build_mask(left_offsets, right_offsets)
    if mask is None:
        return None
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
    block[mask] = int(body_color)
    _assign_edge_colors(block, "left", left, bg)
    _assign_edge_colors(block, "right", right, bg)
    if np.any(block[0, :] != bg) or np.any(block[BLOCK_SIZE - 1, :] != bg):
        return None
    return block


def _render_input_core(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        for block, (r0, c0) in zip(
            scrambled,
            [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)],
        ):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient], BG, SEP)


def _render_legend(width: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    legend = np.full((LEGEND_HEIGHT, width), BG, dtype=int)
    c = 0
    for src, dst in pairs:
        legend[:, c] = int(src)
        legend[:, c + 1] = int(dst)
        c += 3
    return legend


def _wrap_with_legend(core: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    legend = _render_legend(core.shape[1], pairs)
    sep_row = np.full((1, core.shape[1]), SEP, dtype=int)
    return np.vstack([legend, sep_row, core])


def _choose_legend_pairs() -> list[tuple[int, int]]:
    cols = random.sample(PALETTE, 4)
    return [(int(cols[0]), int(cols[1])), (int(cols[2]), int(cols[3]))]


def _choose_link_ids(n_links: int) -> list[int]:
    if n_links == 2:
        return random.choice([[0, 1], [1, 0]])
    return random.choice([[0, 1, 0], [1, 0, 1]])


def _block_body_color(used_colors: set[int]) -> int:
    choices = [color for color in PALETTE if int(color) not in used_colors]
    if not choices:
        choices = list(PALETTE)
    return int(random.choice(choices))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble 5x5 blocks into a strip by legend-guided whole-edge color matching.

    Input structure:
    - bg is the background and sep is the divider color.
    - A 2-row legend sits in the top corner, followed by one full sep row, then several scrambled 5x5 blocks separated by sep lines.
    - Blocks may use multiple colors on the chain-direction edges; the legend columns specify which source edge colors can face which target edge colors.

    Rule:
    - For any candidate neighboring blocks, compare their facing edges as full 5-cell sequences.
    - A connection is valid only when foreground/background occupancy aligns at every offset and every non-bg color pair appears in the legend.
    - Find the only orientation and permutation that make a complete chain, then output the ordered 1xN strip with sep dividers.

    Key insight:
    - Geometry alone is insufficient because many edges share the same occupied offsets; the legend provides a directed color translation that must hold across the entire facing edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)

    sep_rows = _full_sep_rows(g, sep)
    assert sep_rows, "legend separator row missing"
    legend_sep_row = int(sep_rows[0])
    allowed_pairs = _parse_legend(g[:legend_sep_row, :], bg)
    blocks = _split_blocks(g[legend_sep_row + 1 :, :], sep)

    chain_h = _count_orders(blocks, "h", bg, mode="translated", allowed_pairs=allowed_pairs)
    chain_v = _count_orders(blocks, "v", bg, mode="translated", allowed_pairs=allowed_pairs)

    def _first_chain(orient: str) -> tuple[int, ...] | None:
        contacts = [_edge_sequences(block) for block in blocks]
        lead, trail = ("right", "left") if orient == "h" else ("bot", "top")
        for perm in itertools.permutations(range(len(blocks))):
            if all(_match_translated(contacts[perm[i]][lead], contacts[perm[i + 1]][trail], bg, allowed_pairs) for i in range(len(perm) - 1)):
                return tuple(int(v) for v in perm)
        return None

    if chain_h == 1 and chain_v == 0:
        perm = _first_chain("h")
        assert perm is not None
        return _render_strip([blocks[i] for i in perm], "h", bg, sep)
    if chain_v == 1 and chain_h == 0:
        perm = _first_chain("v")
        assert perm is not None
        return _render_strip([blocks[i] for i in perm], "v", bg, sep)
    assert not (chain_h > 0 and chain_v > 0), "ambiguous: both H and V chains found"
    assert chain_h > 0 or chain_v > 0, "must find at least one valid chain"
    raise AssertionError("ambiguous within orientation")


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 4])
    orient = random.choice(["h", "v"])
    n_links = n - 1

    legend_pairs = _choose_legend_pairs()
    allowed_pairs = set(legend_pairs)
    mapping = {int(src): int(dst) for src, dst in legend_pairs}
    offsets = random.choice(EDGE_PATTERNS)
    link_ids = _choose_link_ids(n_links)

    src_colors = (int(legend_pairs[0][0]), int(legend_pairs[1][0]))
    src_sigs = [
        _signature_from_order(offsets, src_colors, BG),
        _signature_from_order(offsets, src_colors[::-1], BG),
    ]
    dst_sigs = [_map_signature(sig, mapping, BG) for sig in src_sigs]

    blocks: list[np.ndarray] = []
    for i in range(n):
        used: set[int] = set()
        kw: dict[str, tuple[int, ...]] = {}
        if orient == "h":
            if i > 0:
                kw["left"] = dst_sigs[link_ids[i - 1]]
                used.update(int(v) for v in kw["left"] if int(v) != BG)
            if i < n - 1:
                kw["right"] = src_sigs[link_ids[i]]
                used.update(int(v) for v in kw["right"] if int(v) != BG)
        else:
            if i > 0:
                kw["top"] = dst_sigs[link_ids[i - 1]]
                used.update(int(v) for v in kw["top"] if int(v) != BG)
            if i < n - 1:
                kw["bottom"] = src_sigs[link_ids[i]]
                used.update(int(v) for v in kw["bottom"] if int(v) != BG)

        block = _make_block(_block_body_color(used), BG, **kw)
        if block is None:
            return None
        blocks.append(block)

    translated_orders = _count_orders(blocks, orient, BG, mode="translated", allowed_pairs=allowed_pairs)
    other_translated = _count_orders(blocks, _OTHER[orient], BG, mode="translated", allowed_pairs=allowed_pairs)
    occupancy_orders = _count_orders(blocks, orient, BG, mode="occupancy")
    identity_orders = _count_orders(blocks, orient, BG, mode="identity")

    if translated_orders != 1 or other_translated != 0:
        return None
    if occupancy_orders <= 1:
        return None
    if identity_orders != 0:
        return None

    out = _render_strip(blocks, orient, BG, SEP)
    inp = _wrap_with_legend(_render_input_core(blocks, orient), legend_pairs)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
