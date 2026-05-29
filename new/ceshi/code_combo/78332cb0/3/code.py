# Combo variant of 78332cb0 (index 3)
# combo_spec: D3/2+D12/1
# Name: 视觉伪装与噪声干扰
#
# Design analysis
# ---------------
# fusion_spec:
#   core_behavior:
#     - 仍然是若干个由 sep=6 分开的 5x5 block，输出仍是按边缘接触顺序重排后的条带。
#     - D3/2: 每个 block 都含 1-2 个干扰色像素；排序时只能看该 block 的“有效主色”边缘接触。
#     - D12/1: 全局背景色不再固定为 7，而是从前景色池中选一个，并令恰好一个 block 的主色=背景色。
#       这个 block 的真实主形状会融入背景，只剩少量噪声可见，因此必须靠 sep 先定位到它的 5x5 区域。
#   concrete_trace:
#     - 原题 `_contacts()` 的隐式前提是“block 内除背景外的像素都属于有效形状”。D3/2 先打破了这一点：
#       角点噪声会制造假的边缘接触，不过滤就会把 block 串错。
#     - D12/1 进一步打破了“有效主色一定是可见的非背景色”这个前提：若某端点 block 的主色=全局背景，
#       那么它的真实接触边在视觉上消失了，旧逻辑既不能从连通分量找到它，也不能从非背景众数恢复它。
#     - 融合后 solver 的执行链变成：
#       1. 先用完整 sep 行/列切出所有 5x5 block，而不是靠前景显著性找对象。
#       2. 对普通 block，仅取出现次数显著最多的非背景色作为主色，忽略 1-2 个噪声像素。
#       3. 对“看起来只有 1-2 个杂点”的伪装 block，不再把这些杂点当主形状，而是把它当成一个
#          “隐藏的链端点”，枚举其可能的单边接触 offset，再与其余可见 block 共同求唯一链。
#     - 一个具体例子：若真实顺序是 A-B-C，其中 C 被背景伪装为几乎空白，A/B 的右左接触 offset 分别为 1/1，
#       B/C 的接触 offset 为 3/3；同时噪声故意让输入顺序形成 A-C-B 的假接触。旧 solver 会把噪声当真，
#       或直接把 C 当成“没有主色的异常块”；新 solver 会先认出 C 是被伪装的端点，再只用 A/B 的主色接触
#       和 C 的候选端点接触恢复 A-B-C。
#   complexity_calibration:
#     - 完全隐形的“中间块”会让链信息过少，人类很难从 2-4 个样例归纳，因此把伪装块限定为链端点。
#       这样可见 block 会先形成一条唯一的可见子链，隐藏块只需补到唯一端点即可，难度上升但不失真。
#     - 真正的主接触继续限制在 interior offset {1,2,3}；噪声优先放在角点 offset {0,4}。
#       这让“哪些边是稳定接触、哪些是零星彩噪”在视觉上可分辨，也让不滤噪的旧逻辑确实会被误导。
#     - 每个 block 的噪声保持 1-2 个，避免视觉过乱；可见主色块仍至少有 4 个同色像素，保证“主色明显多于噪声”。
#   implicit_assumptions_audit:
#     - `get_bg(g)` 仍成立：即使某块主色=背景，整张图的大面积空白仍让全局背景是众数。
#     - `_find_sep()` 仍只依赖完整纯色分隔线；背景改色和噪声都不会破坏整行/整列 sep。
#     - `_split_blocks()` 的前提从“块 visibly non-bg”改成“块由 sep 明确框出”，因此在融合后仍合法。
#     - 原 `_contacts()` 依赖“有效前景可直接由 `block != bg` 得到”，这在融合后彻底失效：
#       可见块要改成“dominant non-bg color only”，伪装块则要改成“候选端点接触枚举”。
#     - 新 solver 新增的隐式假设是：被背景伪装的块是链端点，且其可见非背景像素数 < 4。
#       generator 会显式构造且验证这一点，避免 solver 依赖未被保证的输入。
#   placement_legality:
#     - 所有主形状仍由原题的 5x5 连通 mask 生成器产生，保证单块内不越界、4-连通、且仅在链方向边缘有接触。
#     - 噪声只写到原本的背景单元，不覆盖主形状、不碰 sep、不越出 block；伪装块虽然主色=背景，
#       但噪声仍落在它自己的 5x5 区域内。
#     - 输入仍只采用原题允许的 strip / 2x2 布局，块间只有单格 sep，因此视觉层次保持整洁。
#   visual_observability:
#     - 人类看到的规律是可观察的：大多数块都有一团明显主色和少量异色噪点；另有一个块几乎融进背景，
#       但它被 sep 框出了固定的 5x5 区域，说明“这里也是一个块，只是被背景伪装了”。
#     - 输出不去噪、不改色，只重排 noisy blocks，因此样例会直接提示“噪点不是决定链路的依据”。
#     - 为避免画面杂乱，噪声数量少且多落在角点；sep 始终是整行/整列的单一颜色，能清楚切分区域。
#     - 颜色变化规则也可推断：背景只是从原固定色改成了某个常规前景色，且恰有一个块因此被伪装；
#       不是任意复杂的颜色映射，也不依赖人类不可见属性。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


SEP = 6
COLOR_POOL = [1, 2, 3, 4, 5, 7, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
NOISE_EDGE_OFFSETS = [0, 4]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 1200

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    h, w = a.shape
    for r in range(h):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(w):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[int(r), :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, int(c)] == int(sep)).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if int(r) > int(start):
            segs.append(a[int(start):int(r), :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, int(start):int(c)].copy())
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
            a[:r, int(c + 1):].copy(),
            a[int(r + 1):, :c].copy(),
            a[int(r + 1):, int(c + 1):].copy(),
        ]
    if rows:
        return _split_by_rows(a, sep)
    if cols:
        return _split_by_cols(a, sep)
    return []


def _render_strip(blocks: list[np.ndarray], orient: str, *, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    s = BLOCK_SIZE
    stride = s + 1
    if orient == "h":
        out = np.full((s, n * s + (n - 1)), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + s] = int(sep)
        return out
    out = np.full((n * s + (n - 1), s), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + s, :] = int(sep)
    return out


def _edge_sets_from_mask(mask: np.ndarray) -> dict[str, set[int]]:
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _dominant_visible_color(block: np.ndarray, *, bg: int, sep: int) -> int | None:
    colors = [int(v) for v in block.flatten() if int(v) not in (int(bg), int(sep))]
    if not colors:
        return None
    color, count = Counter(colors).most_common(1)[0]
    if int(count) < FG_AREA_MIN:
        return None
    return int(color)


def _camouflage_contact_candidates(orient: str) -> list[dict[str, set[int]]]:
    base = {"top": set(), "bot": set(), "left": set(), "right": set()}
    out: list[dict[str, set[int]]] = []
    if orient == "h":
        for offset in EDGE_CONTACT_OFFSETS:
            one = {k: set(v) for k, v in base.items()}
            one["left"] = {int(offset)}
            out.append(one)
            two = {k: set(v) for k, v in base.items()}
            two["right"] = {int(offset)}
            out.append(two)
        return out
    for offset in EDGE_CONTACT_OFFSETS:
        one = {k: set(v) for k, v in base.items()}
        one["top"] = {int(offset)}
        out.append(one)
        two = {k: set(v) for k, v in base.items()}
        two["bot"] = {int(offset)}
        out.append(two)
    return out


def _candidate_contacts(block: np.ndarray, orient: str, *, bg: int, sep: int,
                        filter_noise: bool, infer_camouflage: bool) -> list[dict[str, set[int]]]:
    if filter_noise:
        dom = _dominant_visible_color(block, bg=bg, sep=sep)
        if dom is not None:
            return [_edge_sets_from_mask(block == int(dom))]
        if infer_camouflage:
            return _camouflage_contact_candidates(orient)
        return [{"top": set(), "bot": set(), "left": set(), "right": set()}]
    mask = (block != int(bg)) & (block != int(sep))
    if bool(mask.any()):
        return [_edge_sets_from_mask(mask)]
    if infer_camouflage:
        return _camouflage_contact_candidates(orient)
    return [{"top": set(), "bot": set(), "left": set(), "right": set()}]


def _valid_chains(blocks: list[np.ndarray], orient: str, *, bg: int, sep: int,
                  filter_noise: bool, infer_camouflage: bool) -> list[tuple[int, ...]]:
    candidates = [
        _candidate_contacts(block, orient, bg=bg, sep=sep,
                            filter_noise=filter_noise, infer_camouflage=infer_camouflage)
        for block in blocks
    ]
    lead, trail = EDGE_PAIRS[orient]
    orders: set[tuple[int, ...]] = set()

    choice_ranges = [range(len(opts)) for opts in candidates]
    for choice in itertools.product(*choice_ranges):
        contacts = [candidates[i][int(choice[i])] for i in range(len(blocks))]

        def ok(a: int, b: int) -> bool:
            return bool(contacts[int(a)][lead] & contacts[int(b)][trail])

        for perm in itertools.permutations(range(len(blocks))):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
                orders.add(tuple(int(x) for x in perm))
    return sorted(orders)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder sep-delimited 5x5 blocks into the unique strip, even when one endpoint
    block is visually camouflaged into the background and all blocks contain sparse
    noise that must be ignored for matching.

    Input structure:
    - The grid contains 3-4 fixed 5x5 cells separated by full sep-colored rows/columns.
    - Most blocks show one dominant non-background sprite color plus 1-2 small noise pixels.
    - Exactly one block may have its true sprite color equal to the global background, so its
      real shape disappears and only a few distracting non-bg pixels remain visible.

    Rule:
    - First use the sep lines to locate every 5x5 block, including the camouflaged one.
    - For visible blocks, keep only the dominant non-bg color when reading edge contacts.
    - For the camouflaged block, treat it as a hidden chain endpoint and test the few legal
      endpoint contact offsets; choose the only orientation and block order that yields a full chain.
    - Output the original noisy/camouflaged blocks in that chain order, separated by sep lines.

    Key insight:
    - The separators reveal block boundaries even when a block blends into the background, and
      the true chain is recovered by combining those fixed block locations with denoised edge
      contacts from the visible blocks and a small endpoint hypothesis set for the hidden one.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = int(_find_sep(g, bg))
    blocks = _split_blocks(g, sep)
    assert blocks, "must contain separated 5x5 blocks"

    chains_h = _valid_chains(blocks, "h", bg=bg, sep=sep, filter_noise=True, infer_camouflage=True)
    chains_v = _valid_chains(blocks, "v", bg=bg, sep=sep, filter_noise=True, infer_camouflage=True)

    if len(chains_h) == 1 and len(chains_v) == 0:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg=bg, sep=sep)
    if len(chains_v) == 1 and len(chains_h) == 0:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg=bg, sep=sep)
    assert not (chains_h and chains_v), "ambiguous: both orientations valid after camouflage handling"
    assert chains_h or chains_v, "must find at least one valid chain"
    raise AssertionError("ambiguous: multiple chains found in the same orientation")


def _count_orders(blocks: list[np.ndarray], orient: str, *, bg: int, sep: int,
                  filter_noise: bool, infer_camouflage: bool) -> int:
    return len(_valid_chains(
        blocks,
        orient,
        bg=bg,
        sep=sep,
        filter_noise=filter_noise,
        infer_camouflage=infer_camouflage,
    ))


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
    s = BLOCK_SIZE
    mask = np.zeros((s, s), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2):max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:s] = True
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
        if bool(mask[:, 0].any()):
            return None
    elif np.where(mask[:, 0])[0].tolist() != [int(left_offset)]:
        return None

    if right_offset is None:
        if bool(mask[:, s - 1].any()):
            return None
    elif np.where(mask[:, s - 1])[0].tolist() != [int(right_offset)]:
        return None

    if bool(mask[0].any()) or bool(mask[s - 1].any()):
        return None
    if any(bool(mask[r].all()) for r in range(s)):
        return None
    if any(bool(mask[:, c].all()) for c in range(s)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_clean_block(color: int, *, bg: int, left: int | None = None, right: int | None = None,
                      top: int | None = None, bottom: int | None = None) -> np.ndarray | None:
    if top is not None or bottom is not None:
        mask = _base_mask(bottom, top, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(left, right, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(bg), dtype=int)
    block[mask] = int(color)
    return block


def _edge_cell(edge: str, offset: int) -> tuple[int, int]:
    if edge == "left":
        return int(offset), 0
    if edge == "right":
        return int(offset), BLOCK_SIZE - 1
    if edge == "top":
        return 0, int(offset)
    if edge == "bot":
        return BLOCK_SIZE - 1, int(offset)
    raise ValueError(edge)


def _pick_noise_color(block: np.ndarray, *, bg: int) -> int:
    used = set(int(v) for v in block.flatten() if int(v) != int(bg))
    choices = [int(c) for c in COLOR_POOL if int(c) != int(bg) and int(c) not in used]
    if not choices:
        choices = [int(c) for c in COLOR_POOL if int(c) != int(bg)]
    return int(random.choice(choices))


def _pick_noisy_permutation(n: int, hidden_idx: int) -> tuple[int, ...]:
    perms = [
        tuple(int(x) for x in perm)
        for perm in itertools.permutations(range(n))
        if tuple(perm) != tuple(range(n)) and (perm[0] == hidden_idx or perm[-1] == hidden_idx)
    ]
    return random.choice(perms)


def _inject_noise_chain(clean_blocks: list[np.ndarray], orient: str, noisy_perm: tuple[int, ...],
                        *, bg: int) -> list[np.ndarray] | None:
    blocks = [block.copy() for block in clean_blocks]
    if orient == "h":
        out_edge, in_edge = "right", "left"
    else:
        out_edge, in_edge = "bot", "top"

    for a, b in zip(noisy_perm[:-1], noisy_perm[1:]):
        offset = int(random.choice(NOISE_EDGE_OFFSETS))
        ra, ca = _edge_cell(out_edge, offset)
        rb, cb = _edge_cell(in_edge, offset)
        if int(blocks[a][ra, ca]) != int(bg) or int(blocks[b][rb, cb]) != int(bg):
            return None
        blocks[a][ra, ca] = _pick_noise_color(blocks[a], bg=bg)
        blocks[b][rb, cb] = _pick_noise_color(blocks[b], bg=bg)
    return blocks


def _render_input(blocks: list[np.ndarray], output_orient: str, perm: tuple[int, ...], *, bg: int, sep: int) -> np.ndarray:
    scrambled = [blocks[i] for i in perm]
    n = len(scrambled)
    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        s = BLOCK_SIZE
        grid = np.full((2 * s + 1, 2 * s + 1), int(bg), dtype=int)
        grid[s, :] = int(sep)
        grid[:, s] = int(sep)
        slots = [(0, 0), (0, s + 1), (s + 1, 0), (s + 1, s + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, r0, c0)
        return grid
    return _render_strip(scrambled, OTHER[output_orient], bg=bg, sep=sep)


def _noise_only_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = int(_find_sep(g, bg))
    blocks = _split_blocks(g, sep)
    chains_h = _valid_chains(blocks, "h", bg=bg, sep=sep, filter_noise=True, infer_camouflage=False)
    chains_v = _valid_chains(blocks, "v", bg=bg, sep=sep, filter_noise=True, infer_camouflage=False)
    if len(chains_h) == 1 and len(chains_v) == 0:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg=bg, sep=sep)
    if len(chains_v) == 1 and len(chains_h) == 0:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg=bg, sep=sep)
    raise AssertionError("noise-only solver ambiguous or failed")


def _camouflage_only_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = int(_find_sep(g, bg))
    blocks = _split_blocks(g, sep)
    chains_h = _valid_chains(blocks, "h", bg=bg, sep=sep, filter_noise=False, infer_camouflage=True)
    chains_v = _valid_chains(blocks, "v", bg=bg, sep=sep, filter_noise=False, infer_camouflage=True)
    if len(chains_h) == 1 and len(chains_v) == 0:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg=bg, sep=sep)
    if len(chains_v) == 1 and len(chains_h) == 0:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg=bg, sep=sep)
    raise AssertionError("camouflage-only solver ambiguous or failed")


def _camouflaged_block_indices(blocks: list[np.ndarray], *, bg: int, sep: int) -> list[int]:
    out: list[int] = []
    for i, block in enumerate(blocks):
        if _dominant_visible_color(block, bg=bg, sep=sep) is None:
            out.append(int(i))
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    bg = int(random.choice(COLOR_POOL))
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    hidden_idx = random.choice([0, n - 1])

    visible_colors = random.sample([c for c in COLOR_POOL if int(c) != int(bg)], n - 1)
    color_iter = iter(visible_colors)
    lead_kw, trail_kw = ORIENT_KW[orient]

    clean_blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = int(shared_offsets[i - 1])
        if i < n - 1:
            kw[trail_kw] = int(shared_offsets[i])
        color = int(bg) if i == hidden_idx else int(next(color_iter))
        block = _make_clean_block(color, bg=bg, **kw)
        if block is None:
            return None
        clean_blocks.append(block)

    if _count_orders(clean_blocks, orient, bg=bg, sep=SEP, filter_noise=True, infer_camouflage=True) != 1:
        return None
    if _count_orders(clean_blocks, OTHER[orient], bg=bg, sep=SEP, filter_noise=True, infer_camouflage=True) != 0:
        return None

    noisy_perm = _pick_noisy_permutation(n, hidden_idx)
    noisy_blocks = _inject_noise_chain(clean_blocks, orient, noisy_perm, bg=bg)
    if noisy_blocks is None:
        return None

    if _count_orders(noisy_blocks, orient, bg=bg, sep=SEP, filter_noise=True, infer_camouflage=True) != 1:
        return None
    if _count_orders(noisy_blocks, OTHER[orient], bg=bg, sep=SEP, filter_noise=True, infer_camouflage=True) != 0:
        return None

    camo_ids = _camouflaged_block_indices(noisy_blocks, bg=bg, sep=SEP)
    if camo_ids != [int(hidden_idx)]:
        return None

    out = _render_strip(noisy_blocks, orient, bg=bg, sep=SEP)
    inp = _render_input(noisy_blocks, orient, noisy_perm, bg=bg, sep=SEP)
    if not np.array_equal(main(inp), out):
        return None

    try:
        if np.array_equal(_noise_only_main(inp), out):
            return None
    except Exception:
        pass

    try:
        if np.array_equal(_camouflage_only_main(inp), out):
            return None
    except Exception:
        pass
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid fused instance after {MAX_RETRIES} attempts")
