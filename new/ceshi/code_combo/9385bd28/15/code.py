# Combo variant of 9385bd28 (index 15)
# combo_spec: D7/1+D11/3+D16/1
# Name: 移除图例输出+散落像素干扰+多背景色

import random
from collections import Counter
from itertools import combinations

import numpy as np


# Design analysis for combo D7/1 + D11/3 + D16/1
# ------------------------------------------------
# a_concretization_and_trace:
#   - 画面被切成 2-3 条整行横向背景带，例如上方全是颜色 4、中间全是颜色 1、底部全是颜色 8。
#   - 左下角图例仍是竖直两列表，但它位于底部背景带内部；每行给出 (marker_color, fill_color)。
#   - 同一个 marker 颜色可以在多个背景带里各出现一个有效对象；有效对象仍只有两类：
#     两个对角 3 像素 L 角，或一个 2x2 实心块。
#   - 另外还会散落 1-4 个单像素干扰，它们使用图例中的 marker 颜色，但不与任何对象 4 连通。
#   - 小例子：marker=6 在上带是一个 L-pair，在下带又是一个 L-pair；中带再撒一个单独的 6。
#     原始逻辑会先用单一 bg 找 legend，再把所有颜色 6 全部并成一个大 bbox，既跨带又被单点拉偏；
#     融合后应先按“每行多数色”恢复背景带，再在每个背景带里单独找颜色 6 的连通分量，
#     只接受 3 像素角或 2x2 块，忽略那个单点，分别填两个 band-local 菱形，最后把图例擦回各自行背景色。
# b_complexity_calibration:
#   - 背景分区只允许 2-3 条水平整带，不做任意形状分割，保证人眼能直接看到“分区”。
#   - 单像素干扰只保留一种类型，数量限制在 1-4；复杂度集中在“结构优先于颜色”，而不是引入更多杂项。
#   - 每个样例至少强制一个 marker 跨多个背景带复用，避免退化回原题的一色一对象。
# c_implicit_assumption_audit_of_original_main:
#   - `get_bg(grid)` 的“单一全局背景”前提失效，必须改成逐行多数色并压缩成背景带。
#   - `_find_legend(..., bg)` 依赖“非 bg”全局可判定；融合后要改成“非本行背景色”。
#   - `argwhere(g == marker)` 假设同色像素都属于同一个对象；融合后同色像素会跨带复用，还会混入单点干扰，
#     因此必须按背景带切开，再做 4 连通分量分类。
#   - 原始 `out = g.copy()` 会永久保留图例；D7/1 要求最终把 legend cells 恢复为对应行背景色。
#   - 原始 diamond 填充是全局 bbox；融合后必须限制在当前背景带内，避免跨带误填。
# d_placement_legality:
#   - generator 保证所有有效对象完整落在单一背景带内，不跨越带边界。
#   - 同一背景带内，同一 marker 颜色至多有一个有效对象；这样 band-local 结构识别唯一。
#   - 单像素干扰不允许覆盖已有像素，也不允许与任何已放像素 4 邻接，否则会粘连成假结构。
#   - 左下 legend 扫描带左侧被保留，确保从底向上看到的第一个“非行背景色”一定属于 legend。
# e_visual_observability:
#   - 输入里能直接看出大块横向背景带、同色对象在不同带重复出现、以及零散同色单点。
#   - 输出里图例整块消失，但带内填充和 marker 保留；这三个现象叠加后，规则能在少量样例中被归纳。
# f_visual_cleanliness:
#   - 背景带只做整行分区，视觉骨架清楚；对象不跨带，避免同色结构交织成一团。
#   - 单点干扰只占极少数量且不相邻，不会破坏 L 角或 2x2 块的可读性。
# g_color_inference:
#   - marker 颜色不从背景带颜色里选，保证“对象色”和“空白色”可区分。
#   - fill 颜色仍允许等于某条背景带颜色，因此某些填充会隐形；这一点与原题一致。
# h_new_main_assumptions_and_generator_guarantees:
#   - solver 假设每行多数色就是该行背景色；generator 控制每行非背景像素数量远小于网格宽度，并回读校验 bands。
#   - solver 假设 2-3 条背景带是按行连续压缩得到的；generator 只生成整条横带并验证解析结果一致。
#   - solver 假设每个 band-local marker 至多解析出一个有效结构；generator 通过“每带每色至多放一个对象”
#     和单点隔离规则保证这一点。


def _row_backgrounds(g: np.ndarray) -> list[int]:
    """Infer the background color of each row by majority vote."""
    return [int(Counter(map(int, g[r, :])).most_common(1)[0][0]) for r in range(g.shape[0])]


def _find_bands(row_bgs: list[int]) -> list[tuple[int, int, int]]:
    """Compress row-majority colors into contiguous horizontal background bands."""
    assert row_bgs, "grid must have at least one row"
    bands: list[tuple[int, int, int]] = []
    start = 0
    color = int(row_bgs[0])
    for r in range(1, len(row_bgs)):
        if int(row_bgs[r]) != color:
            bands.append((start, r - 1, color))
            start = r
            color = int(row_bgs[r])
    bands.append((start, len(row_bgs) - 1, color))
    return bands


def _find_legend(g: np.ndarray, row_bgs: list[int]) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != row_bgs[r])
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != row_bgs[r])
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == row_bgs[r]:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else int(row_bgs[r])
        entries.append((r, marker, fill))
    return c, entries


def _build_legend_mask(
    shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]
) -> np.ndarray:
    """Boolean mask covering every legend marker/fill cell."""
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _connected_components(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(pts)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _classify_component(comp: set[tuple[int, int]]) -> tuple[str, dict]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    if len(comp) == 1:
        return "single", {"cells": set(comp)}

    if len(comp) == 4 and rmax - rmin == 1 and cmax - cmin == 1:
        expected = {(r, c) for r in (rmin, rmax) for c in (cmin, cmax)}
        if comp == expected:
            return "block", {"cells": set(comp)}

    if len(comp) == 3 and rmax - rmin == 1 and cmax - cmin == 1:
        local = {(r - rmin, c - cmin) for r, c in comp}
        pattern_to_corner = {
            frozenset({(0, 0), (0, 1), (1, 0)}): ("TL", (rmin, cmin)),
            frozenset({(0, 0), (0, 1), (1, 1)}): ("TR", (rmin, cmax)),
            frozenset({(0, 0), (1, 0), (1, 1)}): ("BL", (rmax, cmin)),
            frozenset({(0, 1), (1, 0), (1, 1)}): ("BR", (rmax, cmax)),
        }
        info = pattern_to_corner.get(frozenset(local))
        if info is not None:
            corner, anchor = info
            return "corner", {"cells": set(comp), "corner": corner, "anchor": anchor}

    return "other", {"cells": set(comp)}


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [
            ("TL", rmin, cmin, _L_OFFSETS["TL"]),
            ("BR", rmax, cmax, _L_OFFSETS["BR"]),
        ]
    else:
        anchors = [
            ("TR", rmin, cmax, _L_OFFSETS["TR"]),
            ("BL", rmax, cmin, _L_OFFSETS["BL"]),
        ]
    cells: list[tuple[int, int]] = []
    for _, r, c, offsets in anchors:
        for dr, dc in offsets:
            cells.append((r + dr, c + dc))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids an ambiguous diagonal score tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _build_pair_candidate(a: dict, b: dict) -> dict | None:
    names = {a["corner"], b["corner"]}
    if names == {"TL", "BR"}:
        tl = a if a["corner"] == "TL" else b
        br = a if a["corner"] == "BR" else b
        rmin, cmin = tl["anchor"]
        rmax, cmax = br["anchor"]
        diag = "TLBR"
    elif names == {"TR", "BL"}:
        tr = a if a["corner"] == "TR" else b
        bl = a if a["corner"] == "BL" else b
        rmin, cmin = tr["anchor"][0], bl["anchor"][1]
        rmax, cmax = bl["anchor"][0], tr["anchor"][1]
        diag = "TRBL"
    else:
        return None

    if not (rmin < rmax and cmin < cmax):
        return None
    if not _l_pair_valid_bbox(rmax - rmin + 1, cmax - cmin + 1):
        return None

    cells = set(_l_pair_cells(rmin, cmin, rmax, cmax, diag))
    if cells != a["cells"] | b["cells"]:
        return None

    return {
        "kind": "pair",
        "diag": diag,
        "rmin": rmin,
        "rmax": rmax,
        "cmin": cmin,
        "cmax": cmax,
        "structure_cells": cells,
    }


def _find_marker_structure_in_band(
    g: np.ndarray, legend_mask: np.ndarray, band_mask: np.ndarray, marker: int
) -> dict | None:
    pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask) & band_mask)))
    if not pts:
        return None

    components = _connected_components(pts)
    block_info = None
    corner_infos: list[dict] = []
    for comp in components:
        kind, info = _classify_component(comp)
        if kind == "block":
            assert block_info is None, "multiple 2x2 blocks for one band-local marker are unsupported"
            block_info = {"kind": "block", "cells": info["cells"]}
        elif kind == "corner":
            corner_infos.append(info)

    pair_candidates = []
    for a, b in combinations(corner_infos, 2):
        cand = _build_pair_candidate(a, b)
        if cand is not None:
            pair_candidates.append(cand)

    if block_info is not None and pair_candidates:
        raise AssertionError("band-local marker cannot contain both a block and an L-pair")
    if block_info is not None:
        block_info["all_color_cells"] = set(pts)
        return block_info
    if not pair_candidates:
        return None
    if len(pair_candidates) != 1:
        raise AssertionError("band-local marker must map to exactly one valid L-pair")

    pair = pair_candidates[0]
    pair["all_color_cells"] = set(pts)
    return pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill band-local diamond regions from a bottom-left legend, ignore scattered same-color pixels, then erase the legend.

    Input structure:
    - The canvas is partitioned into 2-3 horizontal background-color bands rather than one global background.
    - A vertical two-column legend still sits near the bottom-left, with each row storing (marker_color, fill_color).
    - In each band, a legend color may form one valid object there: either two opposite 3-pixel L-corners or one 2x2 block; extra isolated pixels of legend colors may also be present.

    Rule:
    - Infer the row background colors, compress them into horizontal bands, and read the legend using row-local background comparison.
    - Process legend entries from bottom to top; inside each band, split same-color pixels into 4-connected components and keep only one valid L-pair or one valid 2x2 block, ignoring scattered single-pixel distractors.
    - Recolor a 2x2 block directly, or for an L-pair fill the opposite diagonal diamond inside that band-local bounding box with the legend fill color.
    - Preserve legend cells and later marker colors while painting overlaps, and after all fills are complete, replace every legend cell with its own row background color.

    Key insight: The legend is only an input-side key, and color alone is no longer a reliable object identifier; you must first respect the visible background partitions, then recover the valid shape structure inside each partition, and only then erase the helper legend from the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_bgs = _row_backgrounds(g)
    bands = _find_bands(row_bgs)
    assert 2 <= len(bands) <= 3, "variant expects 2-3 horizontal background bands"

    legend_c, entries = _find_legend(g, row_bgs)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask((h, w), legend_c, entries)

    band_masks: list[np.ndarray] = []
    for r0, r1, _ in bands:
        band_mask = np.zeros_like(g, dtype=bool)
        band_mask[r0 : r1 + 1, :] = True
        band_masks.append(band_mask)

    structures: dict[tuple[int, int], dict | None] = {}
    marker_order = [marker for _, marker, _ in entries]
    for band_idx, band_mask in enumerate(band_masks):
        for marker in marker_order:
            structures[(band_idx, marker)] = _find_marker_structure_in_band(g, legend_mask, band_mask, marker)

    out = g.copy()
    for i, (_, marker, fill) in enumerate(entries):
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        for band_idx, band_mask in enumerate(band_masks):
            structure = structures[(band_idx, marker)]
            if structure is None:
                continue

            if structure["kind"] == "block":
                block = np.zeros_like(g, dtype=bool)
                for r, c in structure["cells"]:
                    block[r, c] = True
                out[block & (~protect)] = int(fill)
                continue

            local_protect = protect.copy()
            for r, c in structure["all_color_cells"]:
                local_protect[r, c] = True

            rmin = int(structure["rmin"])
            rmax = int(structure["rmax"])
            cmin = int(structure["cmin"])
            cmax = int(structure["cmax"])
            if structure["diag"] == "TLBR":
                a, b = (rmin, cmax), (rmax, cmin)
            else:
                a, b = (rmin, cmin), (rmax, cmax)

            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
            out[diamond & band_mask & (~local_protect)] = int(fill)

    for r in range(h):
        out[r, legend_mask[r]] = int(row_bgs[r])
    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (11, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: background bands ──────────────────────────────────────────
BAND_COUNT_CHOICES = [2, 3]
MIN_BAND_HEIGHT = 3
MIN_BOTTOM_BAND_HEIGHT = 5

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.18
BLOCK_PROB = 0.22
MULTI_BAND_REUSE_PROB = 0.28

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: singleton distractors ─────────────────────────────────────
SINGLETON_COUNT_RANGE = (1, 3)
SINGLETON_RETRIES = 140

# ── CFG: readability guardrails ────────────────────────────────────
MAX_TOTAL_STRUCTURES = 8

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
GEN_RETRIES = 400


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        h_hi = min(BBOX_H_MAX, H)
        w_hi = min(BBOX_W_MAX, W)
        if h_hi < 3 or w_hi < 3:
            return None
        bh = random.randint(3, h_hi)
        bw = random.randint(3, w_hi)
    elif shape_choice == "h2_w_ge4":
        w_hi = min(BBOX_W_MAX, W)
        if H < 2 or w_hi < 4:
            return None
        bh = 2
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4 or W < 2:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _sample_band_heights(H: int, n_bands: int, bottom_min: int) -> list[int] | None:
    """Sample contiguous horizontal band heights."""
    mins = [MIN_BAND_HEIGHT] * n_bands
    mins[-1] = bottom_min
    remaining = H - sum(mins)
    if remaining < 0:
        return None
    heights = mins[:]
    for _ in range(remaining):
        heights[random.randrange(n_bands)] += 1
    return heights


def _build_background(H: int, W: int, n_entries: int):
    """Create the striped background and band metadata."""
    n_bands = random.choice(BAND_COUNT_CHOICES)
    band_heights = _sample_band_heights(H, n_bands, max(n_entries, MIN_BOTTOM_BAND_HEIGHT))
    if band_heights is None:
        return None

    band_colors = random.sample(list(range(10)), n_bands)
    grid = np.zeros((H, W), dtype=int)
    bands: list[tuple[int, int, int]] = []
    r0 = 0
    for height, color in zip(band_heights, band_colors):
        r1 = r0 + height - 1
        grid[r0 : r1 + 1, :] = color
        bands.append((r0, r1, color))
        r0 = r1 + 1
    return grid, bands, set(band_colors)


def _build_legend(grid: np.ndarray, bands: list[tuple[int, int, int]], band_colors: set[int], n_entries: int):
    """Place the legend inside the bottom background band."""
    H, W = grid.shape
    legend_c = random.randint(*LEGEND_COL_RANGE)
    bottom_r0, bottom_r1, bottom_bg = bands[-1]
    legend_bot = bottom_r1
    if legend_bot - n_entries + 1 < bottom_r0:
        return None

    marker_pool = [c for c in range(10) if c not in band_colors]
    if len(marker_pool) < n_entries:
        return None
    marker_colors = random.sample(marker_pool, n_entries)

    fills = [random.randint(0, 9) for _ in range(n_entries)]
    if all(fill == bottom_bg for fill in fills):
        alt = random.choice([c for c in range(10) if c != bottom_bg])
        fills[random.randrange(n_entries)] = alt

    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fills[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    """Protect the left strip so the legend remains the first non-background content from the bottom."""
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _try_place_l_pair_in_band(
    band_r0: int, band_r1: int, W: int, color: int, retries: int, can_place, do_place
) -> bool:
    """Try to place an L-pair fully inside one horizontal band."""
    H_band = band_r1 - band_r0 + 1
    for _ in range(retries):
        size = _pick_bbox_size(H_band, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(band_r0, band_r1 - bh + 1)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _try_place_block_in_band(
    band_r0: int, band_r1: int, W: int, color: int, retries: int, can_place, do_place
) -> bool:
    """Try to place a 2x2 block fully inside one horizontal band."""
    if band_r1 - band_r0 + 1 < 2 or W < 2:
        return False
    for _ in range(retries):
        r0 = random.randint(band_r0, band_r1 - 1)
        c0 = random.randint(0, W - 2)
        cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _choose_active_bands(n_bands: int, force_multi: bool) -> list[int]:
    """Choose which bands receive a non-legend instance of a marker color."""
    if force_multi:
        k = 2 if n_bands == 2 else random.choice([2, 3])
    else:
        if random.random() < MULTI_BAND_REUSE_PROB:
            k = 2 if n_bands == 2 else random.choice([2, 3])
        else:
            k = 1
    return sorted(random.sample(range(n_bands), k))


def _can_place_singleton(H: int, W: int, occupied: np.ndarray, forbidden: np.ndarray, r: int, c: int) -> bool:
    if not (0 <= r < H and 0 <= c < W):
        return False
    if occupied[r, c] or forbidden[r, c]:
        return False
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
            return False
    return True


def _place_singleton_distractors(
    H: int,
    W: int,
    grid: np.ndarray,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    colors: list[int],
) -> int:
    """Place 1-3 isolated single-pixel distractors using legend marker colors."""
    target = random.randint(*SINGLETON_COUNT_RANGE)
    placed = 0
    for i in range(target):
        if i == 0:
            color = colors[0]
        else:
            color = random.choice(colors)
        success = False
        for _ in range(SINGLETON_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if _can_place_singleton(H, W, occupied, forbidden, r, c):
                grid[r, c] = color
                occupied[r, c] = True
                placed += 1
                success = True
                break
        if not success and i == 0:
            return 0
    return placed


def _count_band_local_structures(
    g: np.ndarray,
    bands: list[tuple[int, int, int]],
    entries: list[tuple[int, int, int]],
    legend_mask: np.ndarray,
) -> int:
    """Count valid band-local marker structures to reject overcrowded samples."""
    total = 0
    for r0, r1, _ in bands:
        band_mask = np.zeros_like(g, dtype=bool)
        band_mask[r0 : r1 + 1, :] = True
        for _, marker, _ in entries:
            if _find_marker_structure_in_band(g, legend_mask, band_mask, marker) is not None:
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)

    bg_result = _build_background(H, W, n_entries)
    if bg_result is None:
        return None
    grid, bands, band_colors = bg_result

    legend_result = _build_legend(grid, bands, band_colors, n_entries)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, marker_colors = legend_result
    legend_bot = legend_rows[0][0]

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    forced_multi_idx = random.randrange(len(legend_rows))
    noop_idx = None
    if random.random() < NOOP_PROB and len(legend_rows) > 1:
        candidates = [i for i in range(len(legend_rows)) if i != forced_multi_idx]
        noop_idx = random.choice(candidates)

    active_marker_colors: list[int] = []
    for i, (_, marker, _) in enumerate(legend_rows):
        if i == noop_idx:
            continue

        active_bands = _choose_active_bands(len(bands), force_multi=(i == forced_multi_idx))
        use_block = random.random() < BLOCK_PROB
        for band_idx in active_bands:
            band_r0, band_r1, _ = bands[band_idx]
            if use_block:
                placed = _try_place_block_in_band(
                    band_r0, band_r1, W, marker, BLOCK_RETRIES, can_place, do_place
                )
            else:
                placed = _try_place_l_pair_in_band(
                    band_r0, band_r1, W, marker, LPAIR_RETRIES, can_place, do_place
                )
            if not placed:
                return None
        active_marker_colors.append(marker)

    if not active_marker_colors:
        return None

    forced_multi_marker = legend_rows[forced_multi_idx][1]
    singleton_colors = [forced_multi_marker] + [c for c in active_marker_colors if c != forced_multi_marker]
    placed_singletons = _place_singleton_distractors(H, W, grid, occupied, forbidden, singleton_colors)
    if placed_singletons == 0:
        return None

    row_bgs = _row_backgrounds(grid)
    parsed_bands = _find_bands(row_bgs)
    if len(parsed_bands) != len(bands):
        return None
    if [bg for _, _, bg in parsed_bands] != [bg for _, _, bg in bands]:
        return None

    if _count_band_local_structures(grid, bands, legend_rows, legend_mask) > MAX_TOTAL_STRUCTURES:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    bg_grid = np.repeat(np.array(row_bgs, dtype=int).reshape(H, 1), W, axis=1)
    legend_mask = _build_legend_mask((H, W), legend_c, legend_rows)
    if np.array_equal(out, grid):
        return None
    if not np.all(out[legend_mask] == bg_grid[legend_mask]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
