# Combo variant of 78332cb0 (index 2)
# combo_spec: D1/2+D5/2
# Name: 旋转匹配与累积旋转输出
#
# Design analysis
# ---------------
# fused_spec:
#   preserved:
#     - bg=7, sep=6, 3-4 个 5x5 单色连通块
#     - 输入仍是 strip 或 2x2 grid；输出仍是 1xN strip
#     - 链方向仍只会在 H / V 中唯一成立
#   changed:
#     - 块之间的边缘匹配不再在“输入当前朝向”上直接判断；每个块可先做一次二面体变换
#       （4 种旋转 + 左右翻转后的 4 种旋转）后再参与链恢复
#     - 输出不是“恢复到可匹配朝向后直接排条带”，而是对链中第 k 个恢复块再额外顺时针旋转 90*k 度
#     - 因此规则分成两层：先恢复隐藏的匹配姿态，再叠加可见的链序旋转
#
# a) 具体化与行为追踪
#   - 原题若规范链是水平 [A,B,C]，则只需满足 A.right=B.left, B.right=C.left，然后原样输出 [A,B,C]。
#   - 本融合变体里，先构造规范链 [A,B,C]。输入时把 A 做 flip，B 旋转 90 度，C 保持不变并打散。
#     旧 solver 直接比当前边缘会失败，因为 B 的接触边已经跑到 top/bot。
#   - 新 solver 必须先枚举每块的二面体朝向，找到某组 [A',B',C'] 使它们在某个方向形成唯一链；
#     这一步输出的仍是“恢复匹配姿态”的链。
#   - 但最终答案还不能直接渲染 [A',B',C']，而必须变成 [rot0(A'), rot1(B'), rot2(C')]。
#     差异不只是在链搜索，也在最终渲染：第二层旋转依赖第一层恢复出的链序位置。
#
# b) 复杂度校准
#   - N 保持 3 或 4，单块仍是 5x5，这样 4! * 8^4 量级仍可暴力。
#   - 为避免退化成 D1-only，生成器要求每个规范块都与其 90/180/270 度旋转不同，
#     从而链序旋转会在所有非零位置上产生可见变化。
#   - 为避免退化成原题或“只靠重排就能解”，每个样例至少 2 个块在输入中被改向，
#     且至少 1 个块使用 flip，同时禁止输入当前朝向下在任一方向直接成链。
#   - 接触 offset 仍限制在 interior 的 {1,2,3}，降低 accidental match 导致的歧义。
#   - 允许旋转/翻转后会天然带来反向链等对称候选，因此沿用“总变换量最小”的词典序规则：
#       1. 被改向的块数最少
#       2. 使用 flip 的次数最少
#       3. 90° 旋转步数总和最少
#
# c) 隐式假设审查
#   - _split_blocks / _find_sep 只依赖“5x5 块被完整 sep 行列切开”，融合后仍成立。
#   - 原题 _contacts 默认“当前朝向就是匹配朝向”，融合后失效；所以 solver 必须先生成每块的唯一二面体变体池。
#   - 原题 _find_chain 只枚举 permutation，默认连接只由顺序决定，融合后失效；
#     现在 DFS 需要同时决定“选哪个块”和“该块采用哪个朝向”。
#   - 原题渲染阶段默认“恢复顺序后直接 paste 原块”；融合后失效，
#     因为还必须对链位次 k 叠加 rot_cw(k)。
#   - 一旦允许块重定向，如果不显式检查端点外侧边为空、垂直边为空，
#     就会出现很多假阳性链；因此这些约束必须写进 solver，而不能只依赖 generator 的善意构造。
#
# d) 放置合法性
#   - 所有输入/输出块始终保持 5x5，二面体变换和 90° 旋转都不改变尺寸，因此不会越界。
#   - 2x2 布局仍只用于 n=4；条带布局仍用一条 sep 行/列隔开，不会重叠。
#   - generator 先构造规范链，再独立改向、打散并布局；solver 的切块方式与之严格对应。
#
# e) 视觉可观察性
#   - 人类从样例里能同时看到两件事：
#       1. 输入里有些块显然是歪着/镜像的，必须先“转正”才能接边；
#       2. 输出里排好链后，第 2/3/4 个块又继续被逐步转 90/180/270 度。
#   - 若块对旋转太对称，则第二层规则不可见；若输入里很少有块被改向，则第一层规则不可见。
#     因此生成器分别对这两层做硬约束。
#
# f) 可视化整洁度
#   - 仍保持单色稀疏精灵、固定 5x5 尺寸和单像素 sep 线，不叠加额外标记。
#   - 两层变换都只在块内部重排像素，不引入遮挡，也不改变布局骨架。
#
# g) 变化的可推断性
#   - 不涉及颜色映射；颜色只帮助区分块身份。
#   - 规则完全由形状、边缘接触和链中位置决定，均是人类可直接观察的因素。
#
# h) 量化反退化指标
#   - transform_need = 输入中被改向的块数；要求 >= 2。
#   - flip_need = 输入中通过 flip 才构成最优解的块数；要求 >= 1。
#   - rotated_positions = #{k>0 | rot_cw(k)(canonical_block_k) != canonical_block_k}；要求 = n-1。
#   - no_transform_chain = 输入当前朝向下能否直接成链；要求为 0。
#
# i) 新的 solver 假设与 generator 保证
#   - solver 假设：最优合法链在“最小总变换量”规则下唯一。
#     generator 通过闭环验证 main(input)==expected_output 来筛掉歧义样本。
#   - solver 假设：恢复出的规范链再做位次旋转后，非零位次最好真的可见。
#     generator 通过拒绝所有旋转对称块保证这一点。

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
MAX_RETRIES = 500
MIN_DISTINCT_VARIANTS = 4

_OTHER = {"h": "v", "v": "h"}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.flatten())).most_common(1)[0][0])


def _rotate_cw(block: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(block, k=(-(int(quarter_turns) % 4)))


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
        for is_flip, arr2 in ((False, arr), (True, np.fliplr(arr))):
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


def _rotate_chain(blocks: list[np.ndarray]) -> list[np.ndarray]:
    return [_rotate_cw(block, i) for i, block in enumerate(blocks)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a chain of 5x5 blocks by reorienting them for edge matching, then
    rotate each recovered block again according to its position in the chain.

    Input structure:
    - bg is the most common color; sep is a full-row/full-column divider color.
    - The grid contains 3-4 single-color 5x5 sprite blocks, either in a strip or a 2x2 layout.
    - A block may already be rotated and/or mirrored relative to the hidden chain-forming pose.

    Rule:
    - Choose one dihedral variant of each block so that exactly one horizontal-or-vertical chain is valid.
    - Consecutive blocks must share the same single foreground offset on their facing edges; outer endpoint edges stay empty.
    - After recovering that chain, place the blocks in order and additionally rotate the k-th block clockwise by 90*k.

    Key insight:
    - The task has two transformation layers: first undo each block's hidden orientation to make the edge contacts line up,
    - then apply a second, position-dependent rotation that only becomes knowable after the chain order is fixed.
    """
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
    rotated = _rotate_chain(list(sol["blocks"]))
    return _render_strip(rotated, orient, bg, sep)


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
    block[mask] = int(color)
    if not _is_rotation_distinct(block):
        return None
    return block


def _rich_enough(blocks: list[np.ndarray]) -> bool:
    return all(len(_variant_records(block)) >= MIN_DISTINCT_VARIANTS for block in blocks)


def _choose_transformed_blocks(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, int]] | None:
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

    changed_target = min(random.choice([2, min(3, n)]), len(changeable))
    for _ in range(40):
        changed = set(random.sample(changeable, changed_target))
        usable_flip = sorted(changed & set(flip_changeable))
        if not usable_flip:
            continue
        forced_flip = int(random.choice(usable_flip))

        visible: list[np.ndarray] = []
        transformed_blocks = 0
        flip_blocks = 0
        for i, recs in enumerate(variant_lists):
            if i == forced_flip:
                candidates = [rec for rec in recs if bool(rec["flip"]) and tuple(rec["score"]) != (0, 0, 0)]
            elif i in changed:
                candidates = [rec for rec in recs if tuple(rec["score"]) != (0, 0, 0)]
            else:
                candidates = [rec for rec in recs if tuple(rec["score"]) == (0, 0, 0)]
            if not candidates:
                break
            rec = random.choice(candidates)
            visible.append(np.array(rec["arr"], dtype=int).copy())
            transformed_blocks += int(tuple(rec["score"]) != (0, 0, 0))
            flip_blocks += int(bool(rec["flip"]) and tuple(rec["score"]) != (0, 0, 0))

        if len(visible) == n and transformed_blocks >= 2 and flip_blocks >= 1:
            return visible, {"transformed_blocks": int(transformed_blocks), "flip_blocks": int(flip_blocks)}
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
    canonical_blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = int(shared_offsets[i - 1])
        if i < n - 1:
            kw[trail_kw] = int(shared_offsets[i])
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        canonical_blocks.append(block)

    if not _rich_enough(canonical_blocks):
        return None

    canonical_h = _solve_orientation(canonical_blocks, "h", allow_transforms=False)
    canonical_v = _solve_orientation(canonical_blocks, "v", allow_transforms=False)
    if orient == "h":
        if len(canonical_h) != 1 or canonical_v:
            return None
    else:
        if len(canonical_v) != 1 or canonical_h:
            return None

    rotated_blocks = _rotate_chain(canonical_blocks)
    changed_positions = sum(
        int(not np.array_equal(block, rotated))
        for block, rotated in zip(canonical_blocks[1:], rotated_blocks[1:])
    )
    if changed_positions != n - 1:
        return None

    transformed = _choose_transformed_blocks(canonical_blocks)
    if transformed is None:
        return None
    transformed_blocks, stats = transformed

    visible_blocks, inp = _arrange_input_blocks(transformed_blocks, orient)
    if _solve_orientation(visible_blocks, "h", allow_transforms=False):
        return None
    if _solve_orientation(visible_blocks, "v", allow_transforms=False):
        return None

    out = _render_strip(rotated_blocks, orient, BG, SEP)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    stats["n_blocks"] = int(n)
    stats["output_orient_h"] = int(orient == "h")
    stats["changed_positions"] = int(changed_positions)
    stats["no_transform_chain"] = 0
    return inp, stats


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
