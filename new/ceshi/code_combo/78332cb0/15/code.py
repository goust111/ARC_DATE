# Combo variant of 78332cb0 (index 15)
# combo_spec: D1/2+D4/2+D16/0
# Name: 旋转环形反向输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# Design analysis
# ---------------
# spec_variant:
#   preserved:
#     - 背景色仍为 7，分隔色仍为 6；输入仍由 3-4 个 5x5 单色连通块组成。
#     - 输入布局仍是被整行/整列 sep 分开的 strip 或 2x2 grid；solver 仍先切块。
#     - 每个连接仍只靠链方向两侧边缘的单个 fg offset 相等来判断。
#   changed:
#     - D1/2: 每个块在参与匹配前都允许做一次二面体变换，所以 solver 必须同时恢复顺序和朝向。
#     - D4/2: 不再是开链，而是闭环；每个块在链方向的两侧边都必须各有一个接触点，最后一块还要接回第一块。
#     - D16/0: 找到闭环后，不沿链方向输出，而是从展开锚点开始沿反方向走，并把整条环展开成反轴条带。
#
# a) 具体化与行为追踪
#   - 原题若存在水平开链 A->B->C，则 A.left 为空、C.right 为空，只需排列 [A,B,C] 后水平输出。
#   - 本融合版若存在水平闭环 A->B->C->A，则三块都同时有 left/right 两个接触点，且输入里的 A/B/C 还可能先被旋转或翻转。
#   - solver 新流程：
#       1. 从输入切出若干 5x5 块；
#       2. 对每块枚举全部唯一二面体外观，并只保留“目标方向两侧都有 singleton 接触、垂直两边为空”的朝向；
#       3. DFS 搜索能让相邻边 offset 全部一致、且最后一块能接回第一块的闭环；
#       4. 把闭环按“链方向两端接触 offset 对字典序最小”的块作为锚点展开；
#       5. 从该锚点沿反方向输出，并把每块旋转 90 度后按相反轴渲染成 strip。
#   - 例子：若恢复后的水平闭环锚点顺序是 [A,B,C,D]，则输出次序不是 [A,B,C,D]，而是 [A,D,C,B]；
#     同时每块逆时针旋转 90 度，这样原本用于 left/right 闭环连接的边会变成 vertical strip 的 top/bottom 接缝。
#
# b) 复杂度校准
#   - N 仍限制在 3 或 4，单块仍是 5x5，因此“排列 * 变换”的暴力搜索仍可控。
#   - 闭环天然丢失端点，若不加控制会出现“任意切开都行”的循环歧义；这里把展开锚点固定为
#     链方向接触对 `(incoming_offset, outgoing_offset)` 字典序唯一最小的块，并由 generator 保证该最小对唯一。
#   - offset 仍只从 {1,2,3} 里取，且每个样例会重复使用这些少数档位，使“看边缘接触点”在样例中可学习。
#   - 为了让 D1/2 可观察，generator 继续要求至少 2 个块在输入中被改向，且至少 1 个块使用 flip。
#
# c) 隐式假设审查
#   - 原 `_split_blocks` / `_find_sep` 只依赖“5x5 块 + 整行整列分隔线”，融合后仍成立。
#   - 原 `_contacts` 默认当前朝向就是正确朝向；D1/2 使该假设失效，所以必须对每块先枚举二面体变换。
#   - 原 `_find_chain` 假设存在首尾端点；D4/2 使该假设失效，所以“首块 leading 为空 / 尾块 trailing 为空”
#     必须改成“所有块在链方向两侧都恰有一个接触点，且最后一块能接回第一块”。
#   - 原输出直接沿链方向渲染；D16/0 使该假设失效，因此 render 阶段必须改成“锚点固定 + 反向遍历 + 反轴旋转”。
#   - 新的锚点规则引入了新假设：最小接触对必须唯一。generator 会显式构造并验证这一点。
#
# d) 放置合法性
#   - 输入放置仍只使用 5x5 块、单条 sep 分隔行/列、可选 2x2 网格，因此不会越界或互相覆盖。
#   - 输出仍是 1xN 条带；每块旋转或翻转后尺寸仍为 5x5，反轴渲染不会改变合法性。
#   - generator 仍采用 output-first：先造规范闭环，再独立变换块并打乱输入，最后用 main 回验。
#
# e) 视觉可观察性
#   - 闭环效应体现在：恢复后的每个块在链方向两侧都有接触点，不再存在明显端点块。
#   - 旋转/翻转效应体现在：输入中至少 2 块必须“转正”后才能入环，且至少 1 块需要镜像。
#   - 反向输出效应体现在：输出条带的长轴总与闭环方向相反，而且块序是以锚点为首、沿闭环反方向展开。
#   - 锚点仍只依赖可见边缘 offset，不依赖颜色数值等人类不可视因素。
#
# f) 可视化整洁度
#   - 仍只使用单色精灵、背景和 sep 三类像素；没有额外符号层叠。
#   - 闭环信息全部编码在边缘触点上，输出仍是规整条带，不会出现重叠绘制。
#
# g) 变化的可推断性
#   - 本融合版不引入颜色映射；颜色只帮助人眼区分块身份。
#   - generator 通过少量 offset 档位、唯一最小接触对、以及“至少两块改向”的约束，
#     避免样例退化成“看起来只是普通重排”或“环但无法确定从哪里展开”。
#
# variant_metrics_used_after_testing:
#   - transformed_blocks: 输入中实际被旋转/翻转的块数，要求 >= 2。
#   - flip_blocks: 输入中实际使用 flip 的块数，要求 >= 1。
#   - endpointless_blocks: 规范闭环里在链方向两侧都保留 singleton 接触的块数，要求 = N。
#   - axis_inverted: 输出长轴是否与恢复出的闭环方向相反，要求始终为 1。


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

OTHER = {"h": "v", "v": "h"}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}
OUTPUT_ROT_K = {"h": 1, "v": 3}
PAIR_EDGES = {"h": ("left", "right"), "v": ("top", "bot")}
PERP_EDGES = {"h": ("top", "bot"), "v": ("left", "right")}
RING_OFFSET_PATTERNS = {
    3: [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1)],
    4: [(1, 2, 3, 2), (1, 3, 2, 3), (2, 1, 3, 1), (2, 3, 1, 3)],
}


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


def _split_blocks(a: np.ndarray, sep: int, bg: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    cols = _full_sep_cols(a, sep)
    if rows and cols:
        row_stops = [-1] + rows + [int(a.shape[0])]
        col_stops = [-1] + cols + [int(a.shape[1])]
        blocks: list[np.ndarray] = []
        for i in range(len(row_stops) - 1):
            r0 = int(row_stops[i] + 1)
            r1 = int(row_stops[i + 1])
            if r1 <= r0:
                continue
            for j in range(len(col_stops) - 1):
                c0 = int(col_stops[j] + 1)
                c1 = int(col_stops[j + 1])
                if c1 <= c0:
                    continue
                part = a[r0:r1, c0:c1].copy()
                if np.any(part != int(bg)):
                    blocks.append(part)
        return blocks
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


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != int(bg)
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
            rec = {"arr": arr2.copy(), "flip": bool(is_flip), "score": score}
            if key not in best or tuple(score) < tuple(best[key]["score"]):
                best[key] = rec
    return list(best.values())


def _score_add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]), int(a[2] + b[2]))


def _singleton_pair(cont: dict[str, set[int]], orient: str) -> tuple[int, int]:
    inc_edge, out_edge = PAIR_EDGES[orient]
    return (int(next(iter(cont[inc_edge]))), int(next(iter(cont[out_edge]))))


def _normalize_cycle(sol: dict[str, object], orient: str) -> dict[str, object] | None:
    blocks = [np.array(block, dtype=int).copy() for block in sol["blocks"]]
    contacts = list(sol["contacts"])
    pairs = [_singleton_pair(cont, orient) for cont in contacts]
    min_pair = min(pairs)
    starts = [i for i, pair in enumerate(pairs) if tuple(pair) == tuple(min_pair)]
    if len(starts) != 1:
        return None
    s = int(starts[0])
    return {
        "blocks": blocks[s:] + blocks[:s],
        "contacts": contacts[s:] + contacts[:s],
        "pairs": pairs[s:] + pairs[:s],
        "score": tuple(sol["score"]),
    }


def _cycle_key(sol: dict[str, object], orient: str) -> tuple[object, ...]:
    return (orient,) + tuple(block.tobytes() for block in sol["blocks"])


def _solve_orientation(
    blocks: list[np.ndarray],
    orient: str,
    *,
    allow_transforms: bool,
    bg: int,
) -> list[dict[str, object]]:
    incoming, outgoing = PAIR_EDGES[orient]
    perp_a, perp_b = PERP_EDGES[orient]

    pools: list[list[dict[str, object]]] = []
    for block in blocks:
        raw = _variant_records(block) if allow_transforms else [{"arr": block.copy(), "score": (0, 0, 0)}]
        uniq: dict[bytes, dict[str, object]] = {}
        for rec in raw:
            arr = np.array(rec["arr"], dtype=int)
            cont = _edge_sets(arr, bg)
            if cont[perp_a] or cont[perp_b]:
                continue
            if len(cont[incoming]) != 1 or len(cont[outgoing]) != 1:
                continue
            item = {"arr": arr.copy(), "cont": cont, "score": tuple(rec["score"])}
            key = arr.tobytes()
            if key not in uniq or tuple(item["score"]) < tuple(uniq[key]["score"]):
                uniq[key] = item
        pools.append(list(uniq.values()))

    if any(not pool for pool in pools):
        return []

    n = int(len(blocks))
    used = [False] * n
    raw_solutions: list[dict[str, object]] = []

    def dfs(chain: list[tuple[int, dict[str, object]]], total_score: tuple[int, int, int]) -> None:
        pos = int(len(chain))
        if pos == n:
            first_cont = chain[0][1]["cont"]
            last_cont = chain[-1][1]["cont"]
            if last_cont[outgoing] != first_cont[incoming]:
                return
            raw_solutions.append(
                {
                    "blocks": [np.array(item["arr"], dtype=int).copy() for _idx, item in chain],
                    "contacts": [item["cont"] for _idx, item in chain],
                    "score": tuple(total_score),
                }
            )
            return

        for bi in range(n):
            if used[bi]:
                continue
            for item in pools[bi]:
                if pos > 0:
                    prev_cont = chain[-1][1]["cont"]
                    if prev_cont[outgoing] != item["cont"][incoming]:
                        continue
                used[bi] = True
                chain.append((bi, item))
                dfs(chain, _score_add(total_score, tuple(item["score"])))
                chain.pop()
                used[bi] = False

    dfs([], (0, 0, 0))

    canon: dict[tuple[object, ...], dict[str, object]] = {}
    for sol in raw_solutions:
        norm = _normalize_cycle(sol, orient)
        if norm is None:
            continue
        key = _cycle_key(norm, orient)
        if key not in canon or tuple(norm["score"]) < tuple(canon[key]["score"]):
            canon[key] = norm

    return list(canon.values())


def _best_solutions(solutions: list[dict[str, object]]) -> list[dict[str, object]]:
    if not solutions:
        return []
    best_score = min(tuple(sol["score"]) for sol in solutions)
    return [sol for sol in solutions if tuple(sol["score"]) == tuple(best_score)]


def _reverse_from_anchor(blocks: list[np.ndarray]) -> list[np.ndarray]:
    if not blocks:
        return []
    return [blocks[0].copy()] + [block.copy() for block in blocks[:0:-1]]


def _render_ring_output(blocks: list[np.ndarray], chain_orient: str, bg: int, sep: int) -> np.ndarray:
    reversed_blocks = _reverse_from_anchor(blocks)
    rotated = [np.rot90(block, OUTPUT_ROT_K[chain_orient]).copy() for block in reversed_blocks]
    return _render_strip(rotated, OTHER[chain_orient], bg, sep)


def _best_cycle(blocks: list[np.ndarray], *, allow_transforms: bool, bg: int) -> tuple[str, dict[str, object]]:
    candidates: list[tuple[tuple[int, int, int], str, dict[str, object]]] = []
    for orient in ("h", "v"):
        sols = _best_solutions(_solve_orientation(blocks, orient, allow_transforms=allow_transforms, bg=bg))
        for sol in sols:
            candidates.append((tuple(sol["score"]), orient, sol))
    assert candidates, "must find at least one valid transformed ring"
    best_score = min(score for score, _orient, _sol in candidates)
    best = [(orient, sol) for score, orient, sol in candidates if tuple(score) == tuple(best_score)]
    assert len(best) == 1, "ambiguous: multiple minimum-cost transformed rings found"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a transformable closed ring of 5x5 blocks, then unwrap it in the opposite direction.

    Input structure:
    - Background is the dominant color and separator lines are full rows or columns of a second color.
    - The grid contains 3 or 4 scrambled 5x5 monochrome sprite blocks, separated by single separator lines.
    - Each visible block may already be rotated or flipped relative to the hidden canonical ring.

    Rule:
    - Choose block orientations so that exactly one axis forms a closed ring: every block has one incoming and one outgoing edge-contact on that axis, perpendicular edges stay empty, and the last block reconnects to the first.
    - Normalize the recovered ring by picking the unique block whose chain-edge offset pair is lexicographically smallest.
    - Starting from that anchor block, traverse the ring in the reverse direction, rotate every block 90 degrees so the seam direction flips axes, and render the result as a separator-spaced strip.

    Key insight:
    - The task is no longer "find two endpoints"; all blocks are interior to a loop, so the solver must jointly recover dihedral transforms, enforce the closing seam, and use the visible edge-offset pattern itself to decide where the ring should be cut open before reversing it.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep, bg)
    assert len(blocks) in (3, 4), "variant expects 3 or 4 blocks"
    assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks)

    orient, sol = _best_cycle(blocks, allow_transforms=True, bg=bg)
    return _render_ring_output(sol["blocks"], orient, bg, sep)


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


def _canonical_pairs(blocks: list[np.ndarray], orient: str) -> list[tuple[int, int]]:
    return [_singleton_pair(_edge_sets(block, BG), orient) for block in blocks]


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

    changed_target = min(max(2, n - 1), len(changeable))

    for _ in range(60):
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


def _arrange_input_blocks(blocks: list[np.ndarray], chain_orient: str) -> tuple[list[np.ndarray], np.ndarray]:
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

    inp = _render_strip(arranged, chain_orient, BG, SEP)
    return arranged, inp


def _construct_ring_blocks(n: int, orient: str) -> list[np.ndarray] | None:
    offsets = random.choice(RING_OFFSET_PATTERNS[n])
    lead_kw, trail_kw = ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {
            lead_kw: int(offsets[(i - 1) % n]),
            trail_kw: int(offsets[i]),
        }
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)
    return blocks


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    blocks = _construct_ring_blocks(n, orient)
    if blocks is None or not _rich_enough(blocks):
        return None

    canonical_h = _best_solutions(_solve_orientation(blocks, "h", allow_transforms=False, bg=BG))
    canonical_v = _best_solutions(_solve_orientation(blocks, "v", allow_transforms=False, bg=BG))
    if orient == "h":
        if len(canonical_h) != 1 or canonical_v:
            return None
        canonical = canonical_h[0]
    else:
        if len(canonical_v) != 1 or canonical_h:
            return None
        canonical = canonical_v[0]

    pair_list = list(canonical["pairs"])
    if pair_list.count(min(pair_list)) != 1:
        return None

    scrambled = _choose_scrambled_blocks(canonical["blocks"])
    if scrambled is None:
        return None
    transformed_blocks, stats = scrambled

    visible_blocks, inp = _arrange_input_blocks(transformed_blocks, orient)
    if _solve_orientation(visible_blocks, "h", allow_transforms=False, bg=BG):
        return None
    if _solve_orientation(visible_blocks, "v", allow_transforms=False, bg=BG):
        return None

    out = _render_ring_output(canonical["blocks"], orient, BG, SEP)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    stats["n_blocks"] = int(n)
    stats["endpointless_blocks"] = int(
        sum(
            int(len(_edge_sets(block, BG)[PAIR_EDGES[orient][0]]) == 1 and len(_edge_sets(block, BG)[PAIR_EDGES[orient][1]]) == 1)
            for block in canonical["blocks"]
        )
    )
    stats["axis_inverted"] = int(out.shape[0] != inp.shape[0] or out.shape[1] != inp.shape[1] or OTHER[orient] != orient)
    return inp, stats


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
