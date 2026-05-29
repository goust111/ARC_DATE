# Combo variant of e8686506 (index 33)
# combo_spec: D2/0+D4/0+D6/1
# Name: 多色碎片+上下对称+颜色交换
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题: 输入是单色散落碎片；solver 依赖左右镜像和同色不相邻，把它们直接拼回 LR 对称矩形。
#    - 本融合版:
#      1. 每个碎片先变成双色复合块。对任意 support 形状，按同一个 shape-only 规则推断“内填格”：
#         优先取 bbox 真内部；若没有，则取非角格；再不行就固定 carry 1 格。
#         于是 2x2 缺角 triomino 这类块会表现为“2 格外框 + 1 格内填”。
#      2. 输出矩形的支撑几何改成上下镜像。我们在 8x5 矩形上放 4 个位于上半区的小孔块，
#         再向下镜像得到 4 个下半区块；剩余格构成一块保持原位的 frame。
#      3. 输入散布的是 pre-swap 版本: 上半块 A 与下半镜像块 B 形状互为 UD 镜像，但它们的
#         外框色/内填色成对不同。solver 先从 frame 恢复 8 个槽位，再按双色碎片的 support
#         签名识别哪个槽位该填哪一种 piece，重建 pre-board；最后对每个上下镜像 family，
#         用“下半伙伴”的双色对去重涂整对镜像位置。这样仍然体现了颜色来自对偶块，同时最终
#         成图保持严格 UD 对称。
#    - 例子: 若上方某个缺角块在输入里是 `(border=red, fill=blue)`，其下方镜像块是
#      `(border=green, fill=yellow)`，则输出里这一对上下镜像位置都会显示 `(green, yellow)`。
#      人类仍需先识别配对关系，才能看出“上方的双色载荷来自下方伙伴”。
# b) 复杂度校准
#    - 仅做双色而不做换色，容易退化成“整体搬运”；仅做换色而仍单色，又看不出双层交换。
#      这里强制每块都双色、且镜像伙伴的 border/fill 都不同，保证三层推理都真的发生。
#    - support 模板固定为 8x5 / 4 个上半区面积 2-3 的小孔块并向下镜像，得到 8 个非 frame 块
#      + 1 个 frame 块，共 9 块，既足够重复展示 swap family，又不会让槽位重建变得松散。
#    - 颜色自由度控制在 6 色以内；frame 占 2 色，非 frame 复用剩余颜色，避免视觉爆炸。
# c) 隐式假设审查
#    - 原 `components_mask(g == col)` 的单色分块前提已失效，所以 solver 改为按 `g != bg`
#      的 4 连通块提取整块，再由 `_infer_fill_rel_coords` 恢复 border/fill 角色。
#    - 原依赖“镜像位置同色”的传播式 `_assemble` 已失效，因为换色后上下镜像块的输入颜色
#      本来就不同；现在改成“frame 决定槽位 + support 签名决定 pre-board + 最后 swap 颜色”。
#    - 原 frame 检测按颜色找 8 连通件也会失效，因为 frame 自己也是双色；这里一律按
#      `g != bg` 的 8 连通支撑图找 bbox=输出尺寸 且 UD 对称的 1-2 个部件。
#    - 新 solver 的额外假设:
#      1. 所有 piece 都遵守同一个 shape-only 内填规则。
#      2. 非 frame 槽位全部位于上半区的小孔及其向下镜像，不跨越对称轴。
#      3. 每个 oriented support 签名在输入里对应唯一的 `(border, fill)` 颜色对。
#      这些都由 generator 构造保证。
# d) 放置合法性
#    - 输出先构造，再散布输入。散布时先把整个 frame bbox 暂时标成占位符，所有非 frame patch
#      只能落在 bbox 外，且通过 `random_free_location(... padding=1, connectivity=8)` 保证
#      彼此以及与 frame bbox 至少隔 1 格，不重叠、不粘连。
#    - 最后把 frame patch 按原相对坐标贴回，占位区中非 frame 位置恢复为背景，因此不会覆盖外部。
# e) 视觉可观察性
#    - 人类能直接看到: 输入里每块都不是单色，而是“外框+内填”的双色块；输出是上下对称而非左右；
#      同时同形镜像块的两层颜色在输出里互换。三条信号都能从肉眼观察到。
#    - 规则不依赖颜色 id 的数值大小、扫描顺序或任何不可见排序；pairing 完全来自可见的上下镜像几何。
# f) 可视化整洁度
#    - 所有非 frame 块都是 3 格 triomino，间距固定，frame 是单独一大块；输入不会出现多层叠加。
#    - frame 保持在 bbox 内原位，非 frame 全在外部散落，所以“识别哪块是支架、哪块是拼图”很清楚。
# g) 变化的可推断性
#    - swap 不是任意颜色置换，而是“先找上下镜像伙伴，再把下半伙伴的双色载荷投影到整对位置”。
#    - generator 强制至少 2 组镜像 family 同时满足 `border` 与 `fill` 都不同，这样少量样例里
#      就能反复看到“双色整体成对交换”，而不会退化成只换一层或根本没换。
#
# New solver/generator assumption review
# ------------------------------------
# - `main()` 中逐块解析假设由 `generate_input()` 保证:
#   - 所有非背景 4 连通块都恰好是一个真实 piece；因为散布时保持 8 邻域间隔。
#   - 所有 piece 至少 2 格，且 border/fill 两色不同；因为模板只产出 triomino/frame，颜色对强制异色。
#   - frame 在输入中保持输出中的相对位置，且其支撑图是 UD 对称 bbox；因为生成时 frame 不散布。
#   - 每个 slot signature 在输入里有同样数量的拷贝，且颜色对唯一；因为颜色赋值按 signature family 做。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_VALID_DIMS = [(8, 5)]
_H = 8
_W = 5
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400
_N_COLORS = 6
_MIN_SWAP_FAMILIES = 2

_TOP_SLOT_TEMPLATES = [
    (
        ((0, 0), (0, 1)),
        ((0, 3), (0, 4)),
        ((2, 0), (2, 1), (3, 0)),
        ((2, 4), (3, 3), (3, 4)),
    ),
    (
        ((0, 0), (0, 1)),
        ((0, 3), (0, 4)),
        ((2, 0), (3, 0), (3, 1)),
        ((2, 3), (2, 4), (3, 4)),
    ),
]


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _mirror_sig_ud(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, coords = sig
    return h, w, tuple(sorted((h - 1 - int(r), int(c)) for r, c in coords))


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _infer_fill_rel_coords(
    coords: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], str]:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)

    strict = tuple(sorted((r, c) for r, c in coords if 0 < r < h - 1 and 0 < c < w - 1))
    if strict:
        return strict, "strict"

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    fallback = tuple(sorted((r, c) for r, c in coords if (r, c) not in corners))
    if fallback:
        return fallback, "fallback"

    assert len(coords) > 1, "singleton pieces cannot be turned into two-color fragments"
    carry = tuple(sorted(coords))[len(coords) // 2]
    return (carry,), "carry"


def _support_piece_from_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict:
    pts = np.array(sorted((int(r), int(c)) for r, c in cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(pts)
    rel_coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    fill_rel, fill_mode = _infer_fill_rel_coords(rel_coords)
    return {
        "r0": int(r0),
        "c0": int(c0),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "coords": rel_coords,
        "fill_rel": fill_rel,
        "fill_mode": fill_mode,
        "cells": tuple(sorted((int(r), int(c)) for r, c in pts)),
        "sig": _piece_sig(pts),
        "area": len(rel_coords),
    }


def _piece_from_component(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    piece = _support_piece_from_cells([(int(r), int(c)) for r, c in pts])
    r0 = int(piece["r0"])
    c0 = int(piece["c0"])
    rel_coords = piece["coords"]
    fill_set = set(piece["fill_rel"])
    color_at = {(rr, cc): int(grid[r0 + rr, c0 + cc]) for rr, cc in rel_coords}
    border_colors = {color_at[pos] for pos in rel_coords if pos not in fill_set}
    fill_colors = {color_at[pos] for pos in piece["fill_rel"]}

    assert len(border_colors) == 1, "piece border cells must share one outline color"
    assert len(fill_colors) == 1, "piece fill cells must share one fill color"
    border_color = int(next(iter(border_colors)))
    fill_color = int(next(iter(fill_colors)))
    assert border_color != bg and fill_color != bg and border_color != fill_color

    piece["border_color"] = border_color
    piece["fill_color"] = fill_color
    piece["paint"] = tuple(
        sorted(
            (
                rr,
                cc,
                fill_color if (rr, cc) in fill_set else border_color,
            )
            for rr, cc in rel_coords
        )
    )
    return piece


def _paint_piece(board: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], pair: tuple[int, int]) -> None:
    support = _support_piece_from_cells(cells)
    border_color, fill_color = map(int, pair)
    fill_set = set(support["fill_rel"])
    r0 = int(support["r0"])
    c0 = int(support["c0"])
    for rr, cc in support["coords"]:
        board[r0 + rr, c0 + cc] = fill_color if (rr, cc) in fill_set else border_color


def _piece_cell_colors(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    pair: tuple[int, int],
) -> dict[tuple[int, int], int]:
    support = _support_piece_from_cells(cells)
    border_color, fill_color = map(int, pair)
    fill_set = set(support["fill_rel"])
    out: dict[tuple[int, int], int] = {}
    r0 = int(support["r0"])
    c0 = int(support["c0"])
    for rr, cc in support["coords"]:
        out[(r0 + rr, c0 + cc)] = fill_color if (rr, cc) in fill_set else border_color
    return out


def _is_ud_symmetric_bitmap(parts: list[np.ndarray], H: int, W: int) -> bool:
    all_pts = np.concatenate(parts)
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((H, W), dtype=bool)
    for pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[::-1, :])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[np.ndarray]:
    comps8 = components_mask(g != bg, connectivity=8)
    frame_parts: list[np.ndarray] = []
    for i, p1 in enumerate(comps8):
        r0, c0, r1, c1 = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1 - c0 + 1 == W and _is_ud_symmetric_bitmap([p1], H, W):
            frame_parts = [p1]
            break
        for j in range(i + 1, len(comps8)):
            p2 = comps8[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_ud_symmetric_bitmap([p1, p2], H, W):
                frame_parts = [p1, p2]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 UD-symmetric frame parts whose union bbox equals output dims"
    return frame_parts


def _slotize(frame_mask: np.ndarray) -> list[dict]:
    H, W = frame_mask.shape
    assert H % 2 == 0, "UD slotization assumes even output height"
    mid = H // 2
    top_open = ~frame_mask[:mid, :]
    slots: list[dict] = []
    comps = sorted(
        components_mask(top_open, connectivity=4),
        key=lambda pts: tuple(map(int, pts.min(axis=0))),
    )
    for comp in comps:
        top_cells = sorted((int(r), int(c)) for r, c in comp)
        slots.append({"cells": top_cells, "sig": _piece_sig(np.array(top_cells, dtype=int))})
        bot_cells = sorted((H - 1 - int(r), int(c)) for r, c in comp)
        slots.append({"cells": bot_cells, "sig": _piece_sig(np.array(bot_cells, dtype=int))})
    return slots


def _slot_groups(slots: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]]:
    out: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for slot in slots:
        out.setdefault(slot["sig"], []).append(slot["cells"])
    return out


def _slot_pairs(slots: list[dict], H: int) -> list[dict]:
    by_cells = {frozenset(slot["cells"]): slot for slot in slots}
    pairs = []
    for slot in slots:
        if min(r for r, _ in slot["cells"]) >= (H // 2):
            continue
        mirrored = frozenset((H - 1 - int(r), int(c)) for r, c in slot["cells"])
        mate = by_cells.get(mirrored)
        assert mate is not None, "every top slot must have a mirrored bottom slot"
        pairs.append(
            {
                "top_cells": slot["cells"],
                "bot_cells": mate["cells"],
                "top_sig": slot["sig"],
                "bot_sig": mate["sig"],
            }
        )
    return pairs


def _build_pre_from_input(g: np.ndarray, bg: int) -> tuple[np.ndarray, dict]:
    total_area = int((g != int(bg)).sum())
    assert total_area > 0, "grid must contain non-background pieces"
    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_input_mask = np.zeros_like(g, dtype=bool)
    for pts in frame_parts:
        for r, c in pts:
            frame_input_mask[int(r), int(c)] = True

    pre = np.full((H, W), -1, dtype=int)
    all_frame_pts = np.concatenate(frame_parts)
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for pts in frame_parts:
        for r, c in pts:
            pre[int(r) - fr0, int(c) - fc0] = int(g[int(r), int(c)])

    frame_mask = pre != -1
    slots = _slotize(frame_mask)
    slot_groups = _slot_groups(slots)
    slot_pairs = _slot_pairs(slots, H)

    input_groups: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = {}
    non_frame_piece_count = 0
    two_color_piece_count = 0
    for pts in components_mask(g != bg, connectivity=4):
        piece = _piece_from_component(g, pts, bg)
        if all(frame_input_mask[int(r), int(c)] for r, c in piece["cells"]):
            continue
        input_groups.setdefault(piece["sig"], []).append(piece)
        non_frame_piece_count += 1
        two_color_piece_count += 1

    assert non_frame_piece_count == len(slots), "non-frame piece count must match reconstructed slots"

    pair_by_sig: dict[tuple[int, int, tuple[tuple[int, int], ...]], tuple[int, int]] = {}
    for sig, groups in slot_groups.items():
        pieces = input_groups.get(sig, [])
        assert len(pieces) == len(groups), "slot multiplicity must match input multiplicity"
        pairs = {(int(p["border_color"]), int(p["fill_color"])) for p in pieces}
        assert len(pairs) == 1, "each oriented signature must have one unique border/fill pair"
        pair = next(iter(pairs))
        pair_by_sig[sig] = pair
        for cells in groups:
            _paint_piece(pre, cells, pair)

    assert np.all(pre >= 0), "reconstructed pre-swap board must be dense"
    meta = {
        "slot_groups": slot_groups,
        "slot_pairs": slot_pairs,
        "pair_by_sig": pair_by_sig,
        "frame_mask": frame_mask,
        "dims": (H, W),
        "non_frame_piece_count": non_frame_piece_count,
        "two_color_piece_count": two_color_piece_count,
    }
    return pre, meta


def _swap_board(pre: np.ndarray, slot_pairs: list[dict], pair_by_sig: dict) -> np.ndarray:
    out = pre.copy()
    for pair in slot_pairs:
        donor = _piece_cell_colors(pair["bot_cells"], pair_by_sig[pair["bot_sig"]])
        for (r, c), color in donor.items():
            out[int(r), int(c)] = int(color)
            out[pre.shape[0] - 1 - int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a dense rectangle from scattered two-color fragments, then swap the colors of each up/down mirror pair.

    Input structure:
    - The most common color is background.
    - Every non-background 4-connected component is a single composite fragment with two visible layers:
      one outline color and one carried fill color.
    - One or two composite frame fragments stay grouped in their output-relative position and reveal the final bbox.

    Rule:
    - Treat each non-background component as one whole piece, infer its support shape and its border/fill color roles,
      and use total occupied area to recover the output rectangle size.
    - Use the preserved frame to reconstruct the UD-symmetric slot geometry, refill each slot with the matching
      oriented fragment signature to obtain the pre-swap rectangle, then recolor each mirrored slot family with the
      two-layer color payload donated by its lower mirrored partner.

    Key insight:
    - The frame fixes where pieces go, the composite-piece parsing tells which two colors belong to one shape,
      and the final transformation acts on mirror-paired fragments as a color payload swap rather than a second move.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pre, meta = _build_pre_from_input(g, bg)
    out = _swap_board(pre, meta["slot_pairs"], meta["pair_by_sig"])
    assert np.array_equal(out, out[::-1, :]), "final board must be UD-symmetric after swapping"
    return out.astype(int)


def _build_support_template() -> tuple[list[dict], dict]:
    H, W = _VALID_DIMS[0]
    top_slots = [list(cells) for cells in random.choice(_TOP_SLOT_TEMPLATES)]
    open_mask = np.zeros((H, W), dtype=bool)
    slot_pieces: list[dict] = []
    for cells in top_slots:
        for r, c in cells:
            open_mask[int(r), int(c)] = True
        mirrored = sorted((H - 1 - int(r), int(c)) for r, c in cells)
        for r, c in mirrored:
            open_mask[int(r), int(c)] = True
        slot_pieces.append(_support_piece_from_cells(cells))
        slot_pieces.append(_support_piece_from_cells(mirrored))

    frame_cells = sorted((int(r), int(c)) for r in range(H) for c in range(W) if not open_mask[r, c])
    frame_piece = _support_piece_from_cells(frame_cells)
    frame_piece["is_frame"] = True

    frame_mask = np.zeros((H, W), dtype=bool)
    for r, c in frame_piece["cells"]:
        frame_mask[int(r), int(c)] = True
    assert len(components_mask(frame_mask, connectivity=4)) == 1, "frame must remain a single support component"

    slots = _slotize(frame_mask)
    assert len(slots) == len(slot_pieces), "template slotization must recover every non-frame piece"

    pieces = [frame_piece]
    for slot in slots:
        piece = _support_piece_from_cells(slot["cells"])
        piece["is_frame"] = False
        pieces.append(piece)

    meta = {
        "frame_mask": frame_mask,
        "slots": slots,
        "slot_groups": _slot_groups(slots),
        "slot_pairs": _slot_pairs(slots, H),
    }
    return pieces, meta


def _signature_adjacency(slots: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], set]:
    adjacency: dict[tuple[int, int, tuple[tuple[int, int], ...]], set] = {}
    cell_to_slot: dict[tuple[int, int], int] = {}
    for i, slot in enumerate(slots):
        sig = slot["sig"]
        adjacency.setdefault(sig, set())
        for cell in slot["cells"]:
            cell_to_slot[cell] = i

    for i, slot in enumerate(slots):
        sig = slot["sig"]
        for r, c in slot["cells"]:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                j = cell_to_slot.get((rr, cc))
                if j is None or j == i:
                    continue
                other_sig = slots[j]["sig"]
                if other_sig == sig:
                    raise AssertionError("same-signature copies must not 4-touch in the template")
                adjacency[sig].add(other_sig)
    return adjacency


def _assign_signature_pairs(
    slot_groups: dict,
    adjacency: dict,
    palette: list[int],
    frame_pair: tuple[int, int],
) -> dict | None:
    signatures = sorted(slot_groups)
    for sig in signatures:
        msig = _mirror_sig_ud(sig)
        if msig not in slot_groups:
            return None
        if len(slot_groups[sig]) != len(slot_groups[msig]):
            return None

    asym_families = []
    seen = set()
    for sig in signatures:
        msig = _mirror_sig_ud(sig)
        if sig == msig or sig in seen or msig in seen:
            continue
        seen.add(sig)
        seen.add(msig)
        asym_families.append((sig, msig))
    if len(asym_families) < _MIN_SWAP_FAMILIES:
        return None

    border_pool = [c for c in palette if c not in set(frame_pair)]
    if len(border_pool) < 3:
        return None
    border_by_sig: dict = {}

    def border_domain(sig) -> list[int]:
        msig = _mirror_sig_ud(sig)
        blocked = {border_by_sig[n] for n in adjacency.get(sig, set()) if n in border_by_sig}
        out = []
        for col in border_pool:
            if col in blocked:
                continue
            if msig != sig and msig in border_by_sig and border_by_sig[msig] == col:
                continue
            out.append(col)
        random.shuffle(out)
        return out

    def choose_border():
        cands = [sig for sig in signatures if sig not in border_by_sig]
        if not cands:
            return None
        cands.sort(key=lambda sig: (len(border_domain(sig)), -len(adjacency.get(sig, set()))))
        return cands[0]

    def dfs_border() -> bool:
        sig = choose_border()
        if sig is None:
            return True
        for col in border_domain(sig):
            border_by_sig[sig] = int(col)
            if dfs_border():
                return True
            border_by_sig.pop(sig)
        return False

    if not dfs_border():
        return None

    fill_by_sig: dict = {}
    fill_pool = palette[:]

    def fill_domain(sig) -> list[int]:
        msig = _mirror_sig_ud(sig)
        out = []
        for col in fill_pool:
            if col == border_by_sig[sig]:
                continue
            if msig != sig and msig in fill_by_sig and fill_by_sig[msig] == col:
                continue
            out.append(col)
        random.shuffle(out)
        return out

    def choose_fill():
        cands = [sig for sig in signatures if sig not in fill_by_sig]
        if not cands:
            return None
        cands.sort(key=lambda sig: len(fill_domain(sig)))
        return cands[0]

    def dfs_fill() -> bool:
        sig = choose_fill()
        if sig is None:
            return True
        for col in fill_domain(sig):
            fill_by_sig[sig] = int(col)
            if dfs_fill():
                return True
            fill_by_sig.pop(sig)
        return False

    if not dfs_fill():
        return None

    pair_by_sig = {sig: (int(border_by_sig[sig]), int(fill_by_sig[sig])) for sig in signatures}
    swap_families = sum(
        1
        for sig, msig in asym_families
        if pair_by_sig[sig][0] != pair_by_sig[msig][0] and pair_by_sig[sig][1] != pair_by_sig[msig][1]
    )
    if swap_families < _MIN_SWAP_FAMILIES:
        return None
    return pair_by_sig


def _build_pre_and_out() -> tuple[np.ndarray | None, np.ndarray | None, int, list[dict] | None, dict]:
    pieces, support_meta = _build_support_template()
    palette = list(range(_N_COLORS))
    random.shuffle(palette)
    frame_pair = (palette[0], palette[1])
    adjacency = _signature_adjacency(support_meta["slots"])
    pair_by_sig = _assign_signature_pairs(support_meta["slot_groups"], adjacency, palette, frame_pair)
    if pair_by_sig is None:
        return None, None, 0, None, {}

    pre_abs = np.full((_H, _W), -1, dtype=int)
    frame_piece = next(piece for piece in pieces if piece["is_frame"])
    _paint_piece(pre_abs, frame_piece["cells"], frame_pair)
    for sig, groups in support_meta["slot_groups"].items():
        pair = pair_by_sig[sig]
        for cells in groups:
            _paint_piece(pre_abs, cells, pair)

    assert np.all(pre_abs >= 0), "pre board must be dense"
    out_abs = _swap_board(pre_abs, support_meta["slot_pairs"], pair_by_sig)
    if not np.array_equal(out_abs, out_abs[::-1, :]):
        return None, None, 0, None, {}
    if np.array_equal(out_abs, out_abs[:, ::-1]):
        return None, None, 0, None, {}

    bg = random.randint(0, 9)
    abstract_colors = sorted(int(c) for c in np.unique(pre_abs))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract_colors):
        return None, None, 0, None, {}
    cmap = {ac: available[i] for i, ac in enumerate(abstract_colors)}
    pre = np.vectorize(lambda x: cmap[int(x)])(pre_abs).astype(int)
    out = np.vectorize(lambda x: cmap[int(x)])(out_abs).astype(int)
    meta = {
        "pair_by_sig_abs": pair_by_sig,
        "swap_families": sum(
            1
            for sig in pair_by_sig
            if sig < _mirror_sig_ud(sig)
            and pair_by_sig[sig][0] != pair_by_sig[_mirror_sig_ud(sig)][0]
            and pair_by_sig[sig][1] != pair_by_sig[_mirror_sig_ud(sig)][1]
        ),
        "distinct_colors": len(np.unique(out)),
    }
    return pre, out, bg, pieces, meta


def _patch_from_piece(grid: np.ndarray, piece: dict, bg: int) -> np.ndarray:
    patch = np.full((int(piece["h"]), int(piece["w"])), bg, dtype=int)
    for r, c in piece["cells"]:
        patch[int(r) - int(piece["r0"]), int(c) - int(piece["c0"])] = int(grid[int(r), int(c)])
    return patch


def _scatter(pre: np.ndarray, bg: int, pieces: list[dict]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = pre.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    non_frame = [p for p in pieces if not p["is_frame"]]
    random.shuffle(non_frame)
    for piece in sorted(non_frame, key=lambda p: (-int(p["area"]), random.random())):
        patch = _patch_from_piece(pre, piece, bg)
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = bg
    for piece in [p for p in pieces if p["is_frame"]]:
        patch = _patch_from_piece(pre, piece, bg)
        paste(canvas, patch, off_r + int(piece["r0"]), off_c + int(piece["c0"]), transparent=bg)
    return canvas


def _fusion_stats_from_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pre, meta = _build_pre_from_input(g, bg)
    out = _swap_board(pre, meta["slot_pairs"], meta["pair_by_sig"])
    comps = components_mask(g != bg, connectivity=4)
    pair_by_sig = meta["pair_by_sig"]
    swap_families = sum(
        1
        for sig in pair_by_sig
        if sig < _mirror_sig_ud(sig)
        and pair_by_sig[sig][0] != pair_by_sig[_mirror_sig_ud(sig)][0]
        and pair_by_sig[sig][1] != pair_by_sig[_mirror_sig_ud(sig)][1]
    )
    changed_cells = int(np.sum(pre != out))
    return {
        "pieces": len(comps),
        "two_color_pieces": sum(1 for pts in comps if len({int(g[r, c]) for r, c in pts}) == 2),
        "swap_families": swap_families,
        "changed_cells": changed_cells,
        "ud_symmetric_output": bool(np.array_equal(out, out[::-1, :])),
    }


def _gen_attempt() -> tuple[np.ndarray | None, dict]:
    pre, out, bg, pieces, meta = _build_pre_and_out()
    if pre is None or out is None or pieces is None:
        return None, {}
    if not (4 <= len(np.unique(out)) <= 6):
        return None, {}
    if meta.get("swap_families", 0) < _MIN_SWAP_FAMILIES:
        return None, {}
    try:
        inp = _scatter(pre, bg, pieces)
        got = main(inp)
    except Exception:
        return None, {}
    if np.array_equal(got, out):
        stats = _fusion_stats_from_input(inp)
        if stats["two_color_pieces"] != stats["pieces"]:
            return None, {}
        if stats["swap_families"] < _MIN_SWAP_FAMILIES:
            return None, {}
        if stats["changed_cells"] <= 0:
            return None, {}
        return inp, {**meta, **stats}
    return None, {}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid, _meta = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
