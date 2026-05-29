# Variant of 78332cb0
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻块的facing边缘在相同偏移处共享前景像素（交集大小=1），形成唯一链
#   -> 旋转/翻转后边缘匹配：每个 5x5 块在放入输出条带前，可先做一次二面体变换
#      （4 种旋转 + 左右翻转后的 4 种旋转），solver 需要同时恢复朝向和顺序。
#
# Design analysis
# ---------------
# spec_variant:
#   preserved:
#     - bg=7, sep=6, 3-4 个 5x5 单色连通块
#     - 输入仍是 strip 或 2x2 grid；输出仍是 1xN strip
#     - 链方向仍然只会在 H / V 二选一中唯一成立
#   changed:
#     - 输入中的每个块都可能已经被旋转/翻转
#     - 输出不是“原样重排”，而是把每个块恢复到能形成唯一链的朝向后再排成条带
#     - 由于“任意变换 + 边缘相等”会天然引入整条链反向的 180° 对称解，
#       需要增加一个仍然可见的选择原则：在所有合法链中，取总变换量最小的那条
#
# a) 具体化与行为追踪
#   - 原题里，若目标是水平链，则块 i 的 right 边和块 i+1 的 left 边在同一 offset 处各有 1 个 fg。
#     块的朝向在输入和输出中相同。
#   - 本变体里，先构造一个“规范朝向”的输出链，再把每个块独立做一次随机二面体变换后放回输入。
#     例如规范链中 A.right={1}, B.left={1}；若输入里 B 先旋转 90 度，旧逻辑只会看到
#     B.top={3}，A.right 与 B.left 不再直接匹配，因此原 solver 在原地比边会失败。
#     新 solver 必须先枚举 B 的 8 个朝向，找到某个 B' 使 B'.left={1}，再继续链搜索。
#   - 差异发生在“块与块的匹配”环节：原题只枚举块顺序；变体同时枚举顺序和朝向。
#
# b) 复杂度校准
#   - N 仍为 3 或 4，单块大小仍为 5x5，这样 4! * 8^4 量级仍可暴力但明显比原题更难。
#   - 为避免变体退化成“看起来没变”，生成器要求每个块至少有 4 个不同的二面体外观，
#     且每个样例至少 2 个块被改向、至少 1 个块使用 flip。
#   - 连接 offset 仍限制在 interior 的 {1,2,3}，避免自由度过高导致歧义。
#   - “最小变换量”按词典序比较：
#       1. 被改向的块数最少
#       2. 使用 flip 的次数最少
#       3. 90° 旋转步数总和最少
#     这样正向解与“整体反向 + 每块再转 180°”会被稳定地区分开。
#
# c) 隐式假设审查（原 main 中与变体相关的部分）
#   - _split_blocks / _find_sep 依赖“5x5 块 + 整行/整列分隔线”，该前提仍成立，无需改。
#   - 原 _contacts 只在当前朝向上取边缘 fg，默认“块朝向已正确”。该前提失效，
#     所以必须对每个块生成全部唯一二面体变体后再取边缘信息。
#   - 原 _find_chain 只遍历 permutation，默认“连接关系只由顺序决定”。该前提失效，
#     现在需要 DFS 同时选择未使用块和该块的一个变体，并比较不同合法链的总代价。
#   - 原输出直接渲染 blocks[i]，默认“输入块就是输出朝向”。该前提失效，
#     现在必须渲染 DFS 选中的变体数组。
#   - 原 solver 没显式检查端点外侧边为空、垂直边为空；原 generator 恰好保证了这一点。
#     允许旋转/翻转后，如果不把这些条件写进 solver，会出现大量假阳性链，所以这里改为显式约束。
#
# d) 放置合法性
#   - 二面体变换保持 5x5 尺寸，因此输入布局的边界和不重叠性质与原题一致。
#   - 2x2 布局仍只用于 n=4；strip 布局仍用单条 sep 行/列隔开。
#   - solver 的分块方式与 generator 的布局方式一一对应，不会出现切块歧义。
#
# e) 视觉可观察性
#   - 若块本身高度对称，旋转/翻转不可见，因此生成器拒绝低变体数块。
#   - 样例内至少 2 个块会被转正，且至少 1 个块需要 flip，
#     这样人类能直接看到“除了重排，还需要转正/镜像”。
#   - 同时输出会保留“总改动最少”的那组朝向；从样例上看，就是把块旋到能接上且尽量少折腾。
#   - 颜色不参与排序或变换判断，只保留为单块身份的可见辅助线索。
#
# f) 可视化整洁度
#   - 仍保留单色块、5x5、单一 sep 线和稀疏前景，避免因加入额外标记而让画面杂乱。
#   - 输出仍是规则 strip，块间只有一条分隔线，不叠加额外符号。
#
# g) 颜色变化可推断性
#   - 本变体不改变颜色语义；颜色只用于区分块，规则完全由形状和边缘接触决定。
#
# new_main_assumptions:
#   - 一个合法解要求：所选变体在目标方向的两条垂直边为空；内部连接边为 singleton；
#     首块外侧 leading 边为空，尾块外侧 trailing 边为空。
#   - 若同一方向存在多条合法链，则取“总变换量最小”的唯一解。
#   - generate_input() 显式构造并验证这些条件，并拒绝“无变换也能解”的退化输入。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


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
MIN_DISTINCT_VARIANTS = 4

_OTHER = {"h": "v", "v": "h"}
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
            segs.append(a[int(start) : int(r), :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, int(start) : int(c)].copy())
        start = int(c + 1)
    return segs


def _split_blocks(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    cols = _full_sep_cols(a, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        return [
            a[:r, :c].copy(),
            a[:r, int(c + 1) :].copy(),
            a[int(r + 1) :, :c].copy(),
            a[int(r + 1) :, int(c + 1) :].copy(),
        ]
    if rows:
        return _split_by_rows(a, sep)
    if cols:
        return _split_by_cols(a, sep)
    return []


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    n = int(len(blocks))
    if orient == "h":
        out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc) : int(cc + BLOCK_SIZE)] = block
            cc += BLOCK_SIZE
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

    out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
    rr = 0
    for i, block in enumerate(blocks):
        out[int(rr) : int(rr + BLOCK_SIZE), :] = block
        rr += BLOCK_SIZE
        if int(i) < int(n - 1):
            out[int(rr), :] = int(sep)
            rr += 1
    return out


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    bbg = int(_most_common(block))
    mask = block != int(bbg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _variant_records(block: np.ndarray) -> list[dict[str, object]]:
    best: dict[bytes, dict[str, object]] = {}
    for rot in range(4):
        arr = np.rot90(block, rot)
        for is_flip, arr2 in [(False, arr), (True, np.fliplr(arr))]:
            key = arr2.tobytes()
            turns = min(int(rot), int((4 - rot) % 4))
            score = (
                int(not np.array_equal(arr2, block)),
                int(bool(is_flip)),
                int(turns),
            )
            rec = {"arr": arr2.copy(), "rot": int(rot), "flip": bool(is_flip), "score": score}
            if key not in best or tuple(score) < tuple(best[key]["score"]):
                best[key] = rec
    return list(best.values())


def _score_add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]), int(a[2] + b[2]))


def _solve_orientation(
    blocks: list[np.ndarray], orient: str, *, allow_transforms: bool
) -> list[dict[str, object]]:
    if orient == "h":
        incoming, outgoing = "left", "right"
        perp_a, perp_b = "top", "bot"
    else:
        incoming, outgoing = "top", "bot"
        perp_a, perp_b = "left", "right"

    pools: list[list[dict[str, object]]] = []
    for block in blocks:
        raw = _variant_records(block) if allow_transforms else [{"arr": block.copy(), "score": (0, 0, 0)}]
        uniq: dict[bytes, dict[str, object]] = {}
        for rec in raw:
            arr = np.array(rec["arr"], dtype=int)
            cont = _edge_sets(arr)
            if cont[perp_a] or cont[perp_b]:
                continue
            item = {"arr": arr.copy(), "cont": cont, "score": tuple(rec["score"])}
            key = arr.tobytes()
            if key not in uniq or tuple(item["score"]) < tuple(uniq[key]["score"]):
                uniq[key] = item
        pools.append(list(uniq.values()))

    n = int(len(blocks))
    used = [False] * n
    solutions: list[dict[str, object]] = []

    def dfs(chain: list[tuple[int, dict[str, object]]], total_score: tuple[int, int, int]) -> None:
        pos = int(len(chain))
        if pos == n:
            solutions.append(
                {
                    "blocks": [np.array(item["arr"], dtype=int).copy() for _idx, item in chain],
                    "score": tuple(total_score),
                }
            )
            return

        for bi in range(n):
            if used[bi]:
                continue
            for item in pools[bi]:
                cont = item["cont"]
                if pos == 0:
                    if cont[incoming]:
                        continue
                else:
                    prev_cont = chain[-1][1]["cont"]
                    if len(prev_cont[outgoing]) != 1:
                        continue
                    if len(cont[incoming]) != 1 or cont[incoming] != prev_cont[outgoing]:
                        continue

                if pos < n - 1:
                    if len(cont[outgoing]) != 1:
                        continue
                else:
                    if cont[outgoing]:
                        continue

                used[bi] = True
                chain.append((bi, item))
                dfs(chain, _score_add(total_score, tuple(item["score"])))
                chain.pop()
                used[bi] = False

    dfs([], (0, 0, 0))
    return solutions


def _best_solutions(solutions: list[dict[str, object]]) -> list[dict[str, object]]:
    if not solutions:
        return []
    best_score = min(tuple(sol["score"]) for sol in solutions)
    return [sol for sol in solutions if tuple(sol["score"]) == best_score]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)
    assert not (not blocks)

    sol_h = _solve_orientation(blocks, "h", allow_transforms=True)
    sol_v = _solve_orientation(blocks, "v", allow_transforms=True)

    assert sol_h or sol_v, "must find at least one valid chain"

    candidates: list[tuple[tuple[int, int, int], str, dict[str, object]]] = []
    for orient, sols in (("h", sol_h), ("v", sol_v)):
        for sol in _best_solutions(sols):
            candidates.append((tuple(sol["score"]), orient, sol))
    best_score = min(score for score, _orient, _sol in candidates)
    best = [(orient, sol) for score, orient, sol in candidates if tuple(score) == tuple(best_score)]
    assert len(best) == 1, "ambiguous: multiple minimum-cost transformed chains found"
    orient, sol = best[0]
    return _render_strip(sol["blocks"], orient, bg, sep)


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
    elif np.where(mask[:, 0])[0].tolist() != [int(left_offset)]:
        return None

    if right_offset is None:
        if mask[:, BLOCK_SIZE - 1].any():
            return None
    elif np.where(mask[:, BLOCK_SIZE - 1])[0].tolist() != [int(right_offset)]:
        return None

    if mask[0, :].any() or mask[BLOCK_SIZE - 1, :].any():
        return None
    if any(mask[r, :].all() for r in range(BLOCK_SIZE)):
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


def _rich_enough(blocks: list[np.ndarray]) -> bool:
    return all(len(_variant_records(block)) >= MIN_DISTINCT_VARIANTS for block in blocks)


def _choose_scrambled_blocks(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, int]] | None:
    n = int(len(blocks))
    variant_lists = [_variant_records(block) for block in blocks]
    changeable = [i for i, recs in enumerate(variant_lists) if any(tuple(rec["score"]) != (0, 0, 0) for rec in recs)]
    flip_changeable = [
        i
        for i, recs in enumerate(variant_lists)
        if any(bool(rec["flip"]) and tuple(rec["score"]) != (0, 0, 0) for rec in recs)
    ]
    if len(changeable) < 2 or not flip_changeable:
        return None

    changed_target = random.choice([2, min(3, n)])
    changed_target = min(changed_target, len(changeable))

    for _ in range(40):
        changed = set(random.sample(changeable, changed_target))
        if not changed & set(flip_changeable):
            continue
        forced_flip = int(random.choice(sorted(changed & set(flip_changeable))))
        scrambled: list[np.ndarray] = []
        transformed_blocks = 0
        flip_blocks = 0
        for i, recs in enumerate(variant_lists):
            if i == forced_flip:
                usable = [rec for rec in recs if bool(rec["flip"]) and tuple(rec["score"]) != (0, 0, 0)]
            elif i in changed:
                usable = [rec for rec in recs if tuple(rec["score"]) != (0, 0, 0)]
            else:
                usable = [rec for rec in recs if tuple(rec["score"]) == (0, 0, 0)]
            if not usable:
                break
            rec = random.choice(usable)
            scrambled.append(np.array(rec["arr"], dtype=int).copy())
            transformed_blocks += int(tuple(rec["score"]) != (0, 0, 0))
            flip_blocks += int(bool(rec["flip"]) and tuple(rec["score"]) != (0, 0, 0))
        if len(scrambled) == n and transformed_blocks >= 2 and flip_blocks >= 1:
            return scrambled, {"transformed_blocks": int(transformed_blocks), "flip_blocks": int(flip_blocks)}
    return None


def _arrange_input_blocks(blocks: list[np.ndarray], output_orient: str) -> tuple[list[np.ndarray], np.ndarray]:
    n = int(len(blocks))
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    arranged = [blocks[i].copy() for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        for block, (r0, c0) in zip(
            arranged,
            [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)],
        ):
            paste(grid, block, r0, c0)
        return arranged, grid

    inp = _render_strip(arranged, _OTHER[output_orient], BG, SEP)
    return arranged, inp


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = int(shared_offsets[i - 1])
        if i < n - 1:
            kw[trail_kw] = int(shared_offsets[i])
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)

    if not _rich_enough(blocks):
        return None

    canonical_h = _solve_orientation(blocks, "h", allow_transforms=False)
    canonical_v = _solve_orientation(blocks, "v", allow_transforms=False)
    if orient == "h":
        if len(canonical_h) != 1 or canonical_v:
            return None
    else:
        if len(canonical_v) != 1 or canonical_h:
            return None

    scrambled = _choose_scrambled_blocks(blocks)
    if scrambled is None:
        return None
    transformed_blocks, stats = scrambled

    visible_blocks, inp = _arrange_input_blocks(transformed_blocks, orient)
    if _solve_orientation(visible_blocks, "h", allow_transforms=False):
        return None
    if _solve_orientation(visible_blocks, "v", allow_transforms=False):
        return None

    out = _render_strip(blocks, orient, BG, SEP)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    stats["n_blocks"] = int(n)
    stats["output_orient_h"] = int(orient == "h")
    return inp, stats


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
