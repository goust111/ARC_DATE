# Variant of 78332cb0
# Dimension: D3 — 源对象结构 (src)
# Change: 5x5的单色精灵块，每块一个连通分量，4-10个前景像素，由品红分隔线分开... → 含干扰像素的精灵块
#
# Design analysis
# ---------------
# variant_spec:
#   core_behavior:
#     - 每个 5x5 block 仍然有一个“有效主形状”：单色、4-连通、4-10 像素。
#     - 另外再加入 1-2 个稀疏干扰像素；这些像素颜色不同于主色，通常落在候选匹配边缘上。
#     - 输出不是去噪图，而是把原始 noisy block 重新排序成条带；只有“判断谁和谁相连”这一步要忽略干扰。
#   concrete_trace:
#     - 原题里 `_contacts()` 先取 block 内众数；由于 5x5 中背景最多，众数就是背景。
#     - 然后旧逻辑用 `block != background` 当作前景，因此主色和干扰色都会一起参与边缘匹配。
#     - 例：若 A 的右边缘在 offset=0 处多了一个噪声像素，B 的左边缘也在 offset=0 有一个噪声像素，
#       旧逻辑就会错误地认为 A 可以连到 B。
#     - 变体逻辑应先找出每个 block 的主色（非背景颜色里出现次数最多者），只用该颜色提取 edge contacts，
#       再按这些 contact 排序；渲染输出时则保留整块 noisy block。
#   complexity_calibration:
#     - 干扰像素只取少量（每块 1-2 个），避免视觉杂乱，也保证“主色是块内显著多数”可由 solver 稳定恢复。
#     - 干扰像素优先放到链方向边缘的角点 offset {0,4}，而真正的主色 contact 仍只落在 interior offset {1,2,3}。
#       这样视觉上能明显看出“边上有零星彩色噪点”，同时旧逻辑确实会被它们误导。
#     - generator 额外构造一条“噪声诱导的错误链”，并把输入顺序设成这条错误链，确保过滤噪声是必要步骤。
#   implicit_assumptions_audit:
#     - `_find_sep()` / `_full_sep_rows()` / `_full_sep_cols()` 只依赖完整分隔线，和 D3 变化无关，保持不变。
#     - `_split_blocks()` 只依赖 5x5 block 被 sep 分开，噪声不出 block 边界，因此仍成立。
#     - 原 `_contacts()` 假设“所有非背景像素都是有效前景”，这在变体中失效，必须改成“只看主色像素”。
#     - 原 `_find_chain()` 默认只需找到一个可行排列；变体里旧逻辑会出现多条 noisy chain。
#       新 solver 仍要求按主色过滤后恰有唯一朝向/唯一排列，因此 generator 必须保证这一点。
#   placement_legality:
#     - 主形状仍由原题的输出优先策略生成，保证 5x5 内、4-连通、不越界。
#     - 干扰像素只写入原本的背景单元，不覆盖主形状，不写到 sep 上，也不越出 5x5 block。
#     - 输入布局仍只使用原题允许的 2x2 grid / strip，并沿用单格 sep。
#   visual_observability:
#     - 人类能直接看到：每块都有一个大面积主色图形，旁边夹着 1-2 个零散异色噪点。
#     - 若不忽略这些噪点，边缘接触会显得“乱连”；而输出只是把 noisy block 本身重新排好顺序，
#       因此“先识别主图形、再忽略杂色”的规则可以从少量样例中观察出来。

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# -- Configuration constants --------------------------------------------------
BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
NOISE_EDGE_OFFSETS = [0, 4]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 800
# -----------------------------------------------------------------------------

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


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


def _render_strip(blocks: list[np.ndarray], orient: str, *, bg: int = BG, sep: int = SEP) -> np.ndarray:
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


def _dominant_fg_color(block: np.ndarray, *, bg: int = BG, sep: int = SEP) -> int:
    colors = [int(v) for v in block.flatten() if int(v) not in (int(bg), int(sep))]
    assert colors, "block must contain at least one non-background pixel"
    return int(Counter(colors).most_common(1)[0][0])


def _edge_sets(block: np.ndarray, *, principal_only: bool, bg: int = BG, sep: int = SEP) -> dict[str, set[int]]:
    if principal_only:
        mask = block == int(_dominant_fg_color(block, bg=bg, sep=sep))
    else:
        mask = (block != int(bg)) & (block != int(sep))
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _valid_chains(blocks: list[np.ndarray], orient: str, *, principal_only: bool,
                  bg: int = BG, sep: int = SEP) -> list[tuple[int, ...]]:
    contacts = [_edge_sets(block, principal_only=principal_only, bg=bg, sep=sep) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    out: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            out.append(tuple(int(x) for x in perm))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 noisy blocks into the unique strip determined by the dominant color
    inside each block. Disturbance pixels are preserved in the output but ignored when
    computing edge matches.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = int(_find_sep(g, bg))
    blocks = _split_blocks(g, sep)
    assert blocks, "must contain separated 5x5 blocks"

    chains_h = _valid_chains(blocks, "h", principal_only=True, bg=bg, sep=sep)
    chains_v = _valid_chains(blocks, "v", principal_only=True, bg=bg, sep=sep)

    if len(chains_h) == 1 and len(chains_v) == 0:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg=bg, sep=sep)
    if len(chains_v) == 1 and len(chains_h) == 0:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg=bg, sep=sep)
    assert not (chains_h and chains_v), "ambiguous: both H and V chains found after noise filtering"
    assert chains_h or chains_v, "must find at least one valid chain after noise filtering"
    raise AssertionError("ambiguous: multiple chains found in the same orientation")


def _count_orders(blocks: list[np.ndarray], orient: str, *, principal_only: bool) -> int:
    return int(len(_valid_chains(blocks, orient, principal_only=principal_only, bg=BG, sep=SEP)))


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
        if mask[:, s - 1].any():
            return None
    elif np.where(mask[:, s - 1])[0].tolist() != [right_offset]:
        return None

    if mask[0].any() or mask[s - 1].any():
        return None
    if any(mask[r].all() for r in range(s)):
        return None
    if any(mask[:, c].all() for c in range(s)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_clean_block(color: int, *, left: int | None = None, right: int | None = None,
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

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
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


def _pick_noise_color(block: np.ndarray) -> int:
    used = set(int(v) for v in block.flatten() if int(v) != BG)
    choices = [int(c) for c in PALETTE if int(c) not in used]
    if not choices:
        choices = [int(c) for c in PALETTE if int(c) != int(_dominant_fg_color(block))]
    return int(random.choice(choices))


def _inject_noise_chain(clean_blocks: list[np.ndarray], orient: str,
                        noisy_perm: tuple[int, ...]) -> list[np.ndarray] | None:
    blocks = [block.copy() for block in clean_blocks]
    if orient == "h":
        out_edge, in_edge = "right", "left"
    else:
        out_edge, in_edge = "bot", "top"

    for a, b in zip(noisy_perm[:-1], noisy_perm[1:]):
        offset = int(random.choice(NOISE_EDGE_OFFSETS))
        ra, ca = _edge_cell(out_edge, offset)
        rb, cb = _edge_cell(in_edge, offset)
        if int(blocks[a][ra, ca]) != BG or int(blocks[b][rb, cb]) != BG:
            return None
        blocks[a][ra, ca] = _pick_noise_color(blocks[a])
        blocks[b][rb, cb] = _pick_noise_color(blocks[b])

    for block in blocks:
        dom = int(_dominant_fg_color(block))
        cnt = Counter(int(v) for v in block.flatten() if int(v) != BG)
        if int(cnt[dom]) <= max(v for k, v in cnt.items() if int(k) != dom):
            return None
    return blocks


def _pick_noisy_permutation(n: int) -> tuple[int, ...]:
    perms = [perm for perm in itertools.permutations(range(n)) if tuple(perm) != tuple(range(n))]
    return tuple(int(x) for x in random.choice(perms))


def _render_input(blocks: list[np.ndarray], output_orient: str, perm: tuple[int, ...]) -> np.ndarray:
    scrambled = [blocks[i] for i in perm]
    n = len(scrambled)

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        s = BLOCK_SIZE
        grid = np.full((2 * s + 1, 2 * s + 1), BG, dtype=int)
        grid[s, :] = SEP
        grid[:, s] = SEP
        slots = [(0, 0), (0, s + 1), (s + 1, 0), (s + 1, s + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient], bg=BG, sep=SEP)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = int(_find_sep(g, bg))
    blocks = _split_blocks(g, sep)
    chains_h = _valid_chains(blocks, "h", principal_only=False, bg=bg, sep=sep)
    chains_v = _valid_chains(blocks, "v", principal_only=False, bg=bg, sep=sep)
    if chains_h and not chains_v:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg=bg, sep=sep)
    if chains_v and not chains_h:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg=bg, sep=sep)
    raise AssertionError("legacy solver ambiguous or failed")


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    clean_blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = int(shared_offsets[i - 1])
        if i < n - 1:
            kw[trail_kw] = int(shared_offsets[i])
        block = _make_clean_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        clean_blocks.append(block)

    if _count_orders(clean_blocks, orient, principal_only=True) != 1:
        return None
    if _count_orders(clean_blocks, _OTHER[orient], principal_only=True) != 0:
        return None

    noisy_perm = _pick_noisy_permutation(n)
    noisy_blocks = _inject_noise_chain(clean_blocks, orient, noisy_perm)
    if noisy_blocks is None:
        return None

    if _count_orders(noisy_blocks, orient, principal_only=True) != 1:
        return None
    if _count_orders(noisy_blocks, _OTHER[orient], principal_only=True) != 0:
        return None
    if tuple(range(n)) not in _valid_chains(
        [noisy_blocks[i] for i in noisy_perm],
        orient,
        principal_only=False,
        bg=BG,
        sep=SEP,
    ):
        return None

    out = _render_strip(noisy_blocks, orient, bg=BG, sep=SEP)
    inp = _render_input(noisy_blocks, orient, noisy_perm)
    if not np.array_equal(main(inp), out):
        return None
    try:
        legacy_out = _legacy_main(inp)
    except Exception:
        legacy_out = None
    if legacy_out is None or np.array_equal(legacy_out, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid noisy instance after {MAX_RETRIES} attempts")
