# Variant of 9385bd28
# Dimension: D11 — 干扰项 (meta)
# Change: 0-1 个干扰 L 对（使用非图例颜色的 L 形角标记），解题者需忽略非图例色的标记... → 散落的单色像素干扰

import random
from collections import Counter
from itertools import combinations

import numpy as np

from arc_utils import get_bg

# Design analysis for the D11 variant
# design_analysis:
#   a_concretization_and_trace:
#     input_view:
#       - 图例仍位于左下，逐行给出 (marker_color, fill_color)。
#       - 有效对象仍只有两类：两端各 3 像素的 L 形对角角标，或一个 2x2 实心块。
#       - 新干扰改成若干散落的单像素，它们使用图例里已经出现过的 marker 颜色。
#       - 这些干扰像素彼此分离，也与同色有效 marker 结构分离；视觉上“颜色正确但结构不对”。
#     traced_difference:
#       - 原逻辑: 对某个 marker 颜色直接收集全部同色像素 -> 用整体 bbox 推断对角与填充区域。
#       - 新逻辑: 先把同色像素拆成 4 连通分量 -> 只接受 3 像素 L 角分量或 2x2 block 分量；单像素分量忽略。
#       - 例子: 某颜色原本有 6 个 L-pair 像素，若再撒入一个同色单点，旧逻辑会把 bbox 拉大甚至断言失败；新逻辑只取两个 L 分量，单点保持原样。
#   b_complexity_calibration:
#     - 干扰保持“单像素、同色、散落”这一种类型，复杂度集中在结构识别而不是多规则分支。
#     - 每题仍只有少量 legend 颜色与 0-1 个 block，避免对象种类膨胀。
#     - 单像素数量限制在 1-4 个，足够肉眼可见，也不会把画面塞满。
#   c_implicit_assumption_audit_of_original_main:
#     - `marker_pts = argwhere(g == marker)` 依赖“同色像素全都属于同一个对象”；变体下不成立，必须改成连通分量筛选。
#     - `rmin/rmax/cmin/cmax` 直接来自全部同色点；变体下会被单点干扰污染，导致错误 bbox。
#     - `protect |= (g == marker)` 在原题只保护有效 marker；变体下也会保护单点干扰。这个前提虽然变化，但结果仍合理：干扰像素应被忽略并保持可见。
#     - 对 future marker 的保护同理，现也会顺带保护未来颜色的单像素干扰；这能保持“忽略而非消失”的视觉效果。
#   d_placement_legality:
#     - 单像素干扰不能放进 legend 扫描保护区，不能越界，不能覆盖已有对象。
#     - 单像素干扰不能与任何已放像素 4 邻接，否则会并入别的结构，破坏“单像素”语义。
#     - 有效 L 对和 2x2 block 的放置约束保持原样，因此 solver 仍能唯一识别。
#   e_visual_observability:
#     - 人类可直接看到：有些图例色只形成完整 L 角，有些地方只是零散同色点，而输出只响应 L 角/2x2，不响应散点。
#     - 该规律不依赖颜色编号大小、排序或隐藏状态，只依赖可见形状。
#   f_visual_cleanliness:
#     - 干扰只用单像素且避免与其他对象相邻，保证画面不会因为粘连变成难辨认的杂形。
#     - 仍保留原题的矩形/菱形填充骨架，因此主体图案稳定。
#   g_color_inference:
#     - 颜色映射仍完全由左下图例给出；单像素干扰不引入新的颜色规则，只是强调“同色不等于有效标记”。
#   h_new_main_assumptions_and_generator_guarantees:
#     - solver 假设每个 legend 颜色至多对应一个有效对象（一个 L-pair 或一个 2x2 block）；generator 保证这一点。
#     - solver 假设有效 L 角总是 3 像素、2x2 局部 bbox；generator 继续按该模板生成。
#     - solver 假设单像素干扰与同色有效对象不 4 连通；generator 的单像素放置规则保证这一点。


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
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
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


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
            return "block", {"cells": set(comp), "bbox": (rmin, rmax, cmin, cmax)}

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


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [
            ("TL", rmin, cmin, [(0, 0), (0, 1), (1, 0)]),
            ("BR", rmax, cmax, [(0, 0), (0, -1), (-1, 0)]),
        ]
    else:
        anchors = [
            ("TR", rmin, cmax, [(0, 0), (0, -1), (1, 0)]),
            ("BL", rmax, cmin, [(0, 0), (0, 1), (-1, 0)]),
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


def _find_marker_structure(g: np.ndarray, legend_mask: np.ndarray, marker: int) -> dict | None:
    pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
    if not pts:
        return None

    components = _connected_components(pts)
    block_info = None
    corner_infos: list[dict] = []
    for comp in components:
        kind, info = _classify_component(comp)
        if kind == "block":
            assert block_info is None, "multiple 2x2 blocks for one marker color are unsupported"
            block_info = {"kind": "block", "cells": info["cells"]}
        elif kind == "corner":
            corner_infos.append(info)

    pair_candidates = []
    for a, b in combinations(corner_infos, 2):
        cand = _build_pair_candidate(a, b)
        if cand is not None:
            pair_candidates.append(cand)

    if block_info is not None and pair_candidates:
        raise AssertionError("marker color has both a block and an L-pair structure")
    if block_info is not None:
        block_info["all_color_cells"] = set(pts)
        return block_info
    if not pair_candidates:
        return None
    if len(pair_candidates) != 1:
        raise AssertionError("marker color must map to exactly one valid L-pair")

    pair = pair_candidates[0]
    pair["all_color_cells"] = set(pts)
    return pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor regions by reading a bottom-left legend and locating valid marker structures of each legend color.

    Input structure:
    - The grid has one dominant background color plus a vertical 2-column legend near the lower-left edge.
    - Each legend row gives (marker color, fill color), and non-legend marker objects elsewhere are either L-corner pairs or solid 2x2 blocks.
    - Extra single pixels may appear in legend marker colors, but they do not form L corners or 2x2 blocks.

    Rule:
    - Process legend entries from bottom to top.
    - For each marker color, split same-color pixels into 4-connected components and keep only a valid L-pair or a valid 2x2 block; ignore isolated single-pixel distractors.
    - Recolor a 2x2 block directly, or for an L-pair fill the opposite diagonal diamond inside its bounding box with the legend fill color.
    - Preserve legend cells, current marker-colored cells, and later marker-colored cells so later entries remain visible.

    Key insight: Color alone is no longer enough; the true marker is the component with the right local structure, so scattered same-color pixels must be ignored when recovering the bounding box and diagonal.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]
    structures = {marker: _find_marker_structure(g, legend_mask, marker) for marker in marker_order}

    for i, (_, marker, fill) in enumerate(entries):
        structure = structures[marker]
        if structure is None:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if structure["kind"] == "block":
            block = np.zeros_like(g, dtype=bool)
            for r, c in structure["cells"]:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

        protect_current = np.zeros_like(g, dtype=bool)
        for r, c in structure["all_color_cells"]:
            protect_current[r, c] = True
        protect |= protect_current

        rmin = int(structure["rmin"])
        rmax = int(structure["rmax"])
        cmin = int(structure["cmin"])
        cmax = int(structure["cmax"])
        if structure["diag"] == "TLBR":
            a, b = (rmin, cmax), (rmax, cmin)
        else:
            a, b = (rmin, cmin), (rmax, cmax)

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: single-pixel distractors ──────────────────────────────────
SINGLETON_DISTRACTOR_PROB = 0.85
SINGLETON_COUNT_RANGE = (1, 4)
SINGLETON_RETRIES = 120

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
GEN_RETRIES = 300

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif shape_choice == "h2_w_ge4":
        bh = 2
        w_hi = min(BBOX_W_MAX, W)
        if w_hi < 4:
            return None
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


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
    marker_colors: list[int],
) -> int:
    if random.random() >= SINGLETON_DISTRACTOR_PROB:
        return 0

    target = random.randint(*SINGLETON_COUNT_RANGE)
    placed = 0
    for _ in range(target):
        color = random.choice(marker_colors)
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
        if not success:
            continue
    return placed


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

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

    for i, (_, m, _) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place):
                return None

    _place_singleton_distractors(H, W, grid, occupied, forbidden, marker_colors)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
