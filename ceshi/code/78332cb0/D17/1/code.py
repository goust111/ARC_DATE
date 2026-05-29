# Variant of 78332cb0
# Dimension: D17 — 符号编码/图例 (symbol_instructions)
# Change: 无符号编码，所有信息直接由块的空间结构和边缘像素位置表达
#   → 图例定义边缘匹配颜色规则；顶部角落图例给出“颜色A可接颜色B”的有向颜色对

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

"""
design_analysis:
  variant_visual:
    - 输入最上方新增 2 行角落图例，宽度固定占左上 5 列：两组 2x2 颜色对，中间隔 1 列背景。
      每组的左列颜色表示“前一个块的 trailing-edge 颜色”，右列颜色表示“后一个块的
      leading-edge 颜色”。例如 [红][蓝] 表示红边像素可以接到蓝边像素。
    - 图例下方有一整行分隔色 SEP；再下方仍然是原题的打散 5x5 块布局（条带或 2x2）。
    - 输出不保留图例，只把块按链顺序排成 1xN 条带，和原题一致。
    - 块仍保持单连通，但不再单色：每块有一个主体色，连接边上的像素改成图例指定颜色。

  behavior_trace:
    - 小例子：图例有 (2->8) 与 (4->1)。A 的右边第 2 行是颜色 2；B 的左边第 2 行是颜色 8；
      C 的左边第 2 行是颜色 1。
    - 原题逻辑只看“同偏移上有没有前景”，会把 A->B 和 A->C 都视为可接。
    - 变体逻辑要求“同偏移 + 颜色对在图例中出现”，因此只有 A->B 合法，A->C 非法。
    - 差异发生在原 `_contacts()` / `_find_chain()` 这一段：原来把边缘压成布尔集合即可，
      现在必须保留每个偏移位置的具体颜色并按图例做有向匹配。

  complexity_calibration:
    - 图例只用 2 组颜色对，避免颜色自由度过大；但两组都会在链中反复使用，保证规则可见。
    - 块数仍为 3 或 4，保留原任务体量。生成器偏向 4 块，并在 4 块时刻意制造“只看偏移会歧义”
      的样本，让 D17 不是装饰而是决定顺序的核心信息。
    - 连接偏移仍限制在内部 3 个档位 {1,2,3}，人眼易读，也和原题的“同偏移接边”保持一致。

  implicit_assumptions_review:
    - 原 `_split_blocks()` 默认整张图只有块阵列；加了顶端图例后，这个前提失效，所以新 solver
      必须先找到第一条整行分隔线，把图例区和块区裁开。
    - 原 `_contacts()` 用 `block != most_common(block)` 把块二值化。单色块时无害，但多色块下会
      把 D17 的关键信息抹掉，所以必须改成“按边记录 offset -> color”。
    - 原 `ok(a,b)` 只检查 facing-edge 偏移交集非空；变体下必须同时检查颜色对是否属于图例。
    - 原 `_find_sep()` 仍然可用，因为图例不产生整列/整行纯色线，唯一新增的整行纯色线就是图例下方
      的 SEP 行，它正好也是我们需要的锚点。

  placement_legality:
    - 生成器仍用 output-first：先造合法链，再打乱为输入。
    - 图例区固定高 2、其下固定 1 行 SEP，所以不会与块区重叠。
    - 块区仍完全复用原题的 5x5 + 1 分隔线布局，因此不会越界，也能被 solver 的分块逻辑稳定切分。
    - 所有块在非链方向的两条边保持为空，避免垂直/水平误连。
    - 当一个块的两侧连接恰好落在同一 offset 时，生成器会改走折线路径，避免把整行铺满后触发
      原合法性检查；这只是可视化适配，不改变 solver 的判定规则。

  visual_observability:
    - 输入中能直接看到两类证据：顶部图例给出颜色对；块边缘上也能看到这些颜色真的落在接缝位置。
    - 4 块样本里会大量出现“多个块在同一偏移上都有边缘像素”的情况，此时只看形状无法确定次序，
      必须参考图例颜色对，D17 变化在 2-3 个样例上是肉眼可见的。

  visual_tidy:
    - 图例独立放在顶部，和块区之间用整行 SEP 隔开，避免叠加。
    - 图例只占左上 5 列，不会污染块切分区域。
    - 每个块最多 3 种颜色（主体色 + 最多两个接边色），既能看出多色，又不至于太乱。

  color_inference:
    - 颜色不用于“隐式排序”；颜色只在图例与块边缘位置这一可见结构里起作用。
    - 图例对数量固定为 2，且每对都是有向的非恒等对，解题者能从少量样例中稳定看出
      “左颜色接右颜色”的规则，而不是猜测任意颜色替换。

  generator_guarantees_for_new_main:
    - 顶部一定有且仅有一条先于块区的整行 SEP，solver 可据此裁掉图例带。
    - 图例总是 2 行高、两组 2x2 颜色对、位于左上角 5 列内。
    - 每个块的每条链方向边最多只有 1 个接触像素；若存在接触，则其 offset 与颜色都由链设计指定。
    - 4 块样本要求旧的“只看 offset”规则在链方向上不再唯一，从而证明图例真的参与判定。
"""


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    def _find_sep(a: np.ndarray) -> int:
        h, w = a.shape
        cand: list[int] = []
        for r in range(h):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != bg:
                cand.append(int(a[r, 0]))
        for c in range(w):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != bg:
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(a.shape[0]) if bool((a[r, :] == sep).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(a.shape[1]) if bool((a[:, c] == sep).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if r > start:
                segs.append(a[start:r, :].copy())
            start = r + 1
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if c > start:
                segs.append(a[:, start:c].copy())
            start = c + 1
        return segs

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = len(blocks)
        if axis == 0:
            out = np.full((n * 5 + (n - 1), 5), bg, dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[rr : rr + 5, :] = block
                rr += 5
                if i < n - 1:
                    out[rr, :] = sep
                    rr += 1
            return out
        out = np.full((5, n * 5 + (n - 1)), bg, dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, cc : cc + 5] = block
            cc += 5
            if i < n - 1:
                out[:, cc] = sep
                cc += 1
        return out

    def _parse_legend(a: np.ndarray) -> set[tuple[int, int]]:
        assert a.shape[0] >= 2, "legend band must be at least 2 rows high"
        band = a[:2, :]
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
        assert pairs, "legend must contain at least one color pair"
        return pairs

    def _split_blocks(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        cols = _full_sep_cols(a)
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
            blocks = _split_by_rows(a)
        elif cols:
            blocks = _split_by_cols(a)
        else:
            blocks = []
        assert blocks, "must find at least one block"
        for block in blocks:
            assert block.shape == (5, 5), f"unexpected block shape {block.shape}"
        return blocks

    def _edge_maps(block: np.ndarray) -> dict[str, dict[int, int]]:
        return {
            "top": {int(c): int(block[0, c]) for c in range(5) if int(block[0, c]) != bg},
            "bot": {int(c): int(block[4, c]) for c in range(5) if int(block[4, c]) != bg},
            "left": {int(r): int(block[r, 0]) for r in range(5) if int(block[r, 0]) != bg},
            "right": {int(r): int(block[r, 4]) for r in range(5) if int(block[r, 4]) != bg},
        }

    def _find_chain(blocks: list[np.ndarray], orient: str, allowed_pairs: set[tuple[int, int]]):
        contacts = [_edge_maps(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                src_edge = contacts[a]["right"]
                dst_edge = contacts[b]["left"]
            else:
                src_edge = contacts[a]["bot"]
                dst_edge = contacts[b]["top"]
            for off in set(src_edge) & set(dst_edge):
                if (int(src_edge[off]), int(dst_edge[off])) in allowed_pairs:
                    return True
            return False

        idx = list(range(len(blocks)))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    sep_rows = _full_sep_rows(g)
    assert sep_rows, "legend separator row missing"
    legend_sep_row = int(sep_rows[0])
    allowed_pairs = _parse_legend(g[:legend_sep_row, :])
    blocks = _split_blocks(g[legend_sep_row + 1 :, :])

    chain_h = _find_chain(blocks, "h", allowed_pairs)
    chain_v = _find_chain(blocks, "v", allowed_pairs)

    if chain_h is not None and chain_v is None:
        return _render_strip([blocks[i] for i in chain_h], axis=1)
    if chain_v is not None and chain_h is None:
        return _render_strip([blocks[i] for i in chain_v], axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


# --------------- generator ---------------

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
LEGEND_HEIGHT = 2
LEGEND_PAIRS = 2
LEGEND_WIDTH = 3 * LEGEND_PAIRS - 1

FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

_OTHER = {"h": "v", "v": "h"}


def _edge_maps(block: np.ndarray) -> dict[str, dict[int, int]]:
    return {
        "top": {int(c): int(block[0, c]) for c in range(BLOCK_SIZE) if int(block[0, c]) != BG},
        "bot": {int(c): int(block[BLOCK_SIZE - 1, c]) for c in range(BLOCK_SIZE) if int(block[BLOCK_SIZE - 1, c]) != BG},
        "left": {int(r): int(block[r, 0]) for r in range(BLOCK_SIZE) if int(block[r, 0]) != BG},
        "right": {int(r): int(block[r, BLOCK_SIZE - 1]) for r in range(BLOCK_SIZE) if int(block[r, BLOCK_SIZE - 1]) != BG},
    }


def _count_orders(
    blocks: list[np.ndarray],
    orient: str,
    allowed_pairs: set[tuple[int, int]] | None,
) -> int:
    contacts = [_edge_maps(block) for block in blocks]

    def ok(a: int, b: int) -> bool:
        if orient == "h":
            src_edge = contacts[a]["right"]
            dst_edge = contacts[b]["left"]
        else:
            src_edge = contacts[a]["bot"]
            dst_edge = contacts[b]["top"]
        shared = set(src_edge) & set(dst_edge)
        if allowed_pairs is None:
            return bool(shared)
        for off in shared:
            if (int(src_edge[off]), int(dst_edge[off])) in allowed_pairs:
                return True
        return False

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

    if left_offset is not None and right_offset is not None and left_offset == right_offset:
        spine = 1 if left_offset != 1 else 3
        mask[left_offset, 0:2] = True
        mask[left_offset, 3:BLOCK_SIZE] = True
        mask[min(left_offset, spine) : max(left_offset, spine) + 1, 1] = True
        mask[min(left_offset, spine) : max(left_offset, spine) + 1, 3] = True
        mask[spine, 1:4] = True
    else:
        if left_offset is not None:
            mask[left_offset, 0:3] = True
            mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
        if right_offset is not None:
            mask[right_offset, 2:BLOCK_SIZE] = True
            mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < target_area:
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
    body_color: int,
    *,
    left: tuple[int, int] | None = None,
    right: tuple[int, int] | None = None,
    top: tuple[int, int] | None = None,
    bottom: tuple[int, int] | None = None,
) -> np.ndarray | None:
    if top is not None or bottom is not None:
        mask = _base_mask(None if bottom is None else bottom[0], None if top is None else top[0], random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(None if left is None else left[0], None if right is None else right[0], random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = body_color

    if left is not None:
        off, col = left
        assert mask[off, 0]
        block[off, 0] = col
    if right is not None:
        off, col = right
        assert mask[off, BLOCK_SIZE - 1]
        block[off, BLOCK_SIZE - 1] = col
    if top is not None:
        off, col = top
        assert mask[0, off]
        block[0, off] = col
    if bottom is not None:
        off, col = bottom
        assert mask[BLOCK_SIZE - 1, off]
        block[BLOCK_SIZE - 1, off] = col
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
        for block, (r0, c0) in zip(scrambled, [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient])


def _render_legend(width: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    legend = np.full((LEGEND_HEIGHT, width), BG, dtype=int)
    c = 0
    for left_col, right_col in pairs:
        legend[:, c] = left_col
        legend[:, c + 1] = right_col
        c += 3
    return legend


def _wrap_with_legend(core: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    legend = _render_legend(core.shape[1], pairs)
    sep_row = np.full((1, core.shape[1]), SEP, dtype=int)
    return np.vstack([legend, sep_row, core])


def _choose_legend_pairs() -> list[tuple[int, int]]:
    cols = random.sample(PALETTE, 4)
    return [(cols[0], cols[1]), (cols[2], cols[3])]


def _choose_link_pair_ids(n_links: int) -> list[int]:
    if n_links == 2:
        return random.choice([[0, 1], [1, 0]])
    return random.choice([[0, 1, 0], [1, 0, 1]])


def _old_rule_needed(blocks: list[np.ndarray], orient: str) -> bool:
    return _count_orders(blocks, orient, None) != 1


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 4])
    orient = random.choice(["h", "v"])
    n_links = n - 1

    legend_pairs = _choose_legend_pairs()
    pair_ids = _choose_link_pair_ids(n_links)
    if n == 4:
        anchor = random.choice(EDGE_CONTACT_OFFSETS)
        shared_offsets = [anchor] * n_links
    else:
        shared_offsets = [random.choice(EDGE_CONTACT_OFFSETS) for _ in range(n_links)]

    blocks: list[np.ndarray] = []
    for i in range(n):
        contacts: list[int] = []
        kw: dict[str, tuple[int, int]] = {}
        if orient == "h":
            if i > 0:
                pair = legend_pairs[pair_ids[i - 1]]
                kw["left"] = (shared_offsets[i - 1], pair[1])
                contacts.append(pair[1])
            if i < n - 1:
                pair = legend_pairs[pair_ids[i]]
                kw["right"] = (shared_offsets[i], pair[0])
                contacts.append(pair[0])
        else:
            if i > 0:
                pair = legend_pairs[pair_ids[i - 1]]
                kw["top"] = (shared_offsets[i - 1], pair[1])
                contacts.append(pair[1])
            if i < n - 1:
                pair = legend_pairs[pair_ids[i]]
                kw["bottom"] = (shared_offsets[i], pair[0])
                contacts.append(pair[0])

        body_choices = [color for color in PALETTE if color not in contacts]
        block = _make_block(random.choice(body_choices), **kw)
        if block is None:
            return None
        blocks.append(block)

    allowed_pairs = set(legend_pairs)
    if _count_orders(blocks, orient, allowed_pairs) != 1:
        return None
    if _count_orders(blocks, _OTHER[orient], allowed_pairs) != 0:
        return None
    if n == 4 and not _old_rule_needed(blocks, orient):
        return None

    out = _render_strip(blocks, orient)
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
