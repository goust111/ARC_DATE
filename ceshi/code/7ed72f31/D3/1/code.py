# Variant of 7ed72f31
# Dimension: D3 — 嵌入式标记与精灵的空间关系 (src)
# Change: 红线构成精灵4连通分量的一条边，载荷（payload）完全在红线的一侧，红线跨度与载荷平行方向的完整范围一致... → 红线与精灵分离但平行

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  variant_core:
    original_embedded_line: |
      红线与载荷属于同一个 4 连通分量，红线正好贴在载荷的一侧边界上。
      solver 可以先按 4 连通分量切块，再在块内部读出红线轴。
    new_detached_line: |
      红线仍是水平或垂直整线，且跨度仍与载荷 bbox 在平行方向上的完整范围一致；
      但红线不再接触载荷，而是位于载荷外侧，二者之间隔着 1-3 个背景像素。
      输入里只有“原始载荷 + 红线”，输出再补出红线另一侧、且带同样间隔的镜像。
  concrete_trace:
    example: |
      设水平红线在第 8 行，列范围 4..7；蓝色载荷 bbox 也覆盖列 4..7，
      位于第 11..13 行，那么中间第 9..10 行是 2 格空隙。
      原题会把贴边载荷镜像到第 3..5 行；本变体则必须镜像到第 3..5 行以外，
      且保留对称的 2 格空隙，即最近镜像行是第 5 行，红线两侧都可见空带。
  complexity_calibration:
    choices:
      - 保留“孤立红点做点反射”这条原规则，不引入额外颜色映射或新标记形状。
      - 每个网格强制至少 2 个 detached-line 例子，让人能从少量样例看出“间距也被保留”。
      - 间距只取 1/2/3 三档，离散且可重复，避免自由度过大。
  implicit_assumption_audit:
    old_main_breakage:
      - "`label_mask(g != bg, connectivity=4)` 原本把红线和载荷合并成同一块；现在会把它们分开。"
      - "`if MARK in vals` 只会在混合分量里发现嵌入式红线；现在 detached 红线会被误当成无 payload 的纯红分量。"
      - "`bbox` 和 `eligible(...)` 依赖红线 bbox 与载荷 bbox 属于同一分量；现在不成立。"
      - "原代码扫整张图按颜色+bbox 反射，在新布局里应改成先显式匹配红线与候选载荷分量，再只反射该分量。"
    new_solver_assumptions:
      - 红线分量必须是单一直线（同一行或同一列），长度至少 2。
      - 候选载荷必须是单色 4 连通分量。
      - 对于水平线，载荷 bbox 的列范围必须与红线完全一致；垂直线同理要求行范围一致。
      - 载荷必须完全处于红线一侧，且与红线之间的背景间距在 1..3。
      - 每条 detached 红线只匹配到一个这样的载荷；generator 用保留 bbox 的放置策略保证这一点。
  placement_legality:
    generator_strategy: |
      继续采用 output-first。先构造“原始载荷 + 红线 + 镜像”的完整输出 sprite，
      再把镜像半边擦掉得到输入。
      为避免其他对象落入红线与载荷之间的空隙，placement 使用 sprite 的整个 bbox 作为保留区，
      不是只保留有色像素。
  visual_observability:
    why_visible:
      - 载荷使用随机不对称连通形状，镜像方向明确可见。
      - 红线与载荷之间的空带在输入和输出里都清楚可见。
      - 同一张图至少两个 detached-line 精灵，且间距在 1/2/3 中取值，便于比较。
  cleanliness:
    measures:
      - sprite 之间用 bbox 保留区 + 1 格 padding 隔开，避免对象插入别人的空隙。
      - payload 单色、红线固定红色、背景单一，视觉层次保持接近原题。
  color_inference:
    note: |
      颜色不参与排序或映射。每个 payload 仍是单色，只做几何镜像，因此解题者只需关注
      “哪一块被哪条平行红线控制，以及空隙宽度也被复制”。
  generator_non_degeneracy_check:
    metric: |
      统计每个样本中 detached-line 数量以及其 gap 分布。若 gap 全部恒为 0 或所有 line
      都与 payload 4 连通，就退化回原题；本生成器会显式生成 gap∈{1,2,3}，并在每图放入
      至少两个 detached-line 精灵。
"""


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 20, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MIN_DETACHED, MAX_DETACHED = 2, 4
MAX_ISOLATED = 3

# Detached-line payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80
DETACHED_GAP_MIN, DETACHED_GAP_MAX = 1, 3

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 400

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _component_records(mask: np.ndarray, grid: np.ndarray, *, connectivity: int = 4) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(grid[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "colors": set(vals),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def _line_axis(red_comp: dict) -> tuple[str, int] | None:
    coords = red_comp["coords"]
    if len(coords) < 2:
        return None
    rs = {int(r) for r, _ in coords}
    cs = {int(c) for _, c in coords}
    if len(rs) == 1:
        return ("row", int(next(iter(rs))))
    if len(cs) == 1:
        return ("col", int(next(iter(cs))))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detached red row/col lines mirror a nearby monochrome payload across that line.
    - The payload is outside the line, parallel to it, and separated by 1-3 background pixels.
    - The mirror preserves that gap on the opposite side.
    - Original isolated red single-pixel point reflections remain unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    color_comps = _component_records((g != int(bg)) & (g != int(MARK)), g, connectivity=4)
    red_comps = _component_records(g == int(MARK), g, connectivity=4)
    mixed_comps = _component_records(g != int(bg), g, connectivity=4)

    # --- 1) Detached red lines mirror one nearby payload component ---
    used_payloads: set[int] = set()
    for red_comp in red_comps:
        axis = _line_axis(red_comp)
        if axis is None:
            continue

        r0, c0, r1, c1 = map(int, red_comp["bbox"])
        matches: list[tuple[int, int]] = []
        for idx, comp in enumerate(color_comps):
            if idx in used_payloads or len(comp["colors"]) != 1:
                continue
            pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
            gap = None
            if axis[0] == "row":
                if pc0 != c0 or pc1 != c1:
                    continue
                if pr1 < int(axis[1]):
                    gap = int(axis[1]) - pr1 - 1
                elif pr0 > int(axis[1]):
                    gap = pr0 - int(axis[1]) - 1
                else:
                    continue
            else:
                if pr0 != r0 or pr1 != r1:
                    continue
                if pc1 < int(axis[1]):
                    gap = int(axis[1]) - pc1 - 1
                elif pc0 > int(axis[1]):
                    gap = pc0 - int(axis[1]) - 1
                else:
                    continue
            if gap is None or not (DETACHED_GAP_MIN <= int(gap) <= DETACHED_GAP_MAX):
                continue
            matches.append((idx, int(gap)))

        if len(matches) != 1:
            continue

        idx, _gap = matches[0]
        used_payloads.add(idx)
        comp = color_comps[idx]
        color = int(comp["vals"][0])
        for r, c in comp["coords"]:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            if inb(rr, cc):
                out[rr, cc] = color

    # --- 2) Preserve the original rare mixed point-marker behavior ---
    for comp in mixed_comps:
        reds = [(int(r), int(c)) for (r, c), v in zip(comp["coords"], comp["vals"]) if int(v) == int(MARK)]
        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(comp["coords"], comp["vals"]) if int(v) != int(MARK)]
        if len(reds) != 1 or not payload:
            continue
        pr, pc = reds[0]
        for r, c, v in payload:
            rr, cc = refl_point((pr, pc), r, c)
            if inb(rr, cc):
                out[rr, cc] = int(v)

    # --- 3) Original isolated red pixels reflect adjacent 8-connected color components ---
    marker_pts = [comp["coords"][0] for comp in red_comps if len(comp["coords"]) == 1]
    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((pr, pc), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_detached_sprite(rng: random.Random, gap: int) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    payload_first = rng.random() < 0.5

    if is_row:
        H, W = 2 * ph + 2 * gap + 1, pw
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        line_r = ph + gap
        top_slice = np.s_[:ph, :]
        bot_slice = np.s_[line_r + gap + 1 :, :]
        if payload_first:
            payload[top_slice] = pmask
            mirror[bot_slice] = pmask[::-1]
        else:
            payload[bot_slice] = pmask
            mirror[top_slice] = pmask[::-1]
        red_pixels = [(line_r, c) for c in range(W)]
        axis = "row"
    else:
        H, W = ph, 2 * pw + 2 * gap + 1
        payload = np.zeros((H, W), dtype=bool)
        mirror = np.zeros((H, W), dtype=bool)
        line_c = pw + gap
        left_slice = np.s_[:, :pw]
        right_slice = np.s_[:, line_c + gap + 1 :]
        if payload_first:
            payload[left_slice] = pmask
            mirror[right_slice] = pmask[:, ::-1]
        else:
            payload[right_slice] = pmask
            mirror[left_slice] = pmask[:, ::-1]
        red_pixels = [(r, line_c) for r in range(H)]
        axis = "col"

    return {
        "kind": "detached",
        "payload": payload,
        "mirror": mirror,
        "red_pixels": red_pixels,
        "placement_mask": np.ones(payload.shape, dtype=bool),
        "gap": int(gap),
        "axis": axis,
    }


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    H, W = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            H = max(all_rs) - minr + 1
            W = max(all_cs) - minc + 1

            payload = np.zeros((H, W), dtype=bool)
            mirror = np.zeros((H, W), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {
                "kind": "isolated",
                "payload": payload,
                "mirror": mirror,
                "red_pixels": [red_pos],
                "placement_mask": np.ones(payload.shape, dtype=bool),
            }
    return None


def _choose_detached_gaps(count: int, rng: random.Random) -> list[int]:
    vals = [1, 2, 3]
    gaps = [rng.choice(vals) for _ in range(count)]
    if count >= 2 and len(set(gaps)) == 1:
        gaps[rng.randrange(count)] = rng.choice([v for v in vals if v != gaps[0]])
    return gaps


def _random_free_location(blocked: np.ndarray, patch_mask: np.ndarray, rng: random.Random):
    H, W = blocked.shape
    h, w = patch_mask.shape
    if h > H or w > W:
        return None
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (blocked[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = bg


def _reserve_sprite(blocked: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["placement_mask"].shape
    reserved = np.zeros_like(blocked, dtype=bool)
    reserved[r0 : r0 + h, c0 : c0 + w] = spr["placement_mask"]
    if PLACEMENT_PADDING > 0:
        reserved = ndimage.binary_dilation(reserved, structure=STRUCT8, iterations=PLACEMENT_PADDING)
    blocked |= reserved


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_detached = rng.randint(max(MIN_DETACHED, total_markers - MAX_ISOLATED), min(MAX_DETACHED, total_markers))
    n_isolated = total_markers - n_detached

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for gap in _choose_detached_gaps(n_detached, rng):
        spr = _build_detached_sprite(rng, gap)
        if spr is None:
            return None
        sprites.append(spr)
    for _ in range(n_isolated):
        spr = _build_isolated_sprite(rng)
        if spr is None:
            return None
        sprites.append(spr)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, c in zip(sprites, assignment):
        spr["color"] = c

    output = np.full((N, N), bg, dtype=int)
    blocked = np.zeros((N, N), dtype=bool)
    rng.shuffle(sprites)
    for spr in sprites:
        loc = _random_free_location(blocked, spr["placement_mask"], rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = r0, c0
        _stamp_sprite(output, spr, r0, c0)
        _reserve_sprite(blocked, spr, r0, c0)

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
