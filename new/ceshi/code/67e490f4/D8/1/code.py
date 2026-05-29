# Variant of 67e490f4
# Dimension: D8 — 目标结构：框架的对称性 (dst)
# Change: 框架内洞的布局具有 LR+UD+rot180 对称性（四重对称），通过在左上象限放置后镜像到其他象限实现... → 仅左右对称（单轴对称）

# Design analysis
# ===============
# concreteization:
# - 原题的可见主体是一个大框架，框里挖出若干“洞”，框外散落一批小精灵。
# - solver 实际只依赖“洞的形状”和“外部精灵的形状-颜色频次映射”，并不依赖洞在框内如何对称。
# - 因此 D8 的核心变化应落在 generator：旧版是在左上象限放一个洞，然后同时镜像到右上、左下、右下，
#   所以上下结构天然重复；新版改成只把左侧洞镜像到右侧，同一行保留左右对应，但不再强制在上下两半复制。
# - 具体小例子：若左上方第 2 行有一个 1x2 横洞，旧逻辑会在第 2 行右侧和倒数第 3 行两侧再放三份；
#   新逻辑只在第 2 行右侧放一份。底部是否再出现同形洞，改为独立决定。
#
# complexity calibration:
# - 仍保留 3-6 种洞形、外部精灵按形状编码颜色、颜色由“同形精灵中的多数色”决定，避免把题目改成别的机制。
# - 为了让“只左右对称”能被肉眼看出来，生成器会拒绝那些仍然上下对称的洞布局。
# - 还额外要求上下半区都至少有洞，并且洞的逐行分布与其上下翻转版本存在可见差异（row_gap），
#   避免样本偶然退化成原题的四重对称外观。
#
# implicit assumptions audit for main():
# - largest monochrome non-bg component == frame：变体后仍成立，因为框架依旧是唯一的大型单色连通体。
# - holes = bbox 内所有非 frame_col 的 4 连通块：这一步只要求“框里有洞”，不要求上下重复，因此仍成立。
# - shape matching allows rotation/flip：变体不改这条，外部精灵仍可旋转/翻转后与洞匹配。
# - tie-break by smaller color id 仍在 solver 中保留；为避免不可见的颜色 id 真正起作用，generator 保证赢家严格多数，不制造平票。
# - 原版 `_validate_positions` 默认同组镜像副本不会互相粘连；LR-only 后左右副本更可能靠近中轴，因此这里必须显式检查
#   “同一次放置产生的组件数 == 位置数”，否则两个左右副本可能合并成一个大洞，solver 会读成错误形状。
#
# placement legality:
# - 洞不能碰到框 bbox 边界，边框必须保持 frame_col。
# - 每个洞形实例仍是 4-连通，任意两个洞之间保持至少一格 frame_col 的 4 邻域隔离。
# - 左右镜像副本之间至少留出一列框颜色，避免它们在中轴附近连成一个组件。
# - 外部精灵仍完全放在框外，并保持 Chebyshev padding，确保 solver 抽取到的是分离的 sprite 组件。
#
# visual cleanliness:
# - 输出仍是裁剪后的框 bbox，主体清晰，外部精灵不进入框内。
# - 同一洞形的多个实例仍共享同一种填充色，便于从少量样例中看出“形状决定颜色键”的机制。
# - 变体的主要视觉差异来自洞的上下分布不再成对重复，而不是来自额外装饰或颜色技巧。
#
# new solver assumptions after modification:
# - 每种洞形在框外至少有一种匹配 sprite；generator 用 oracle 检查 `main(inp) == output` 来保证。
# - 所有洞形都来自连接的小 shape library，且每个放置组都保持为独立组件。
# - 只要 generator 保证上述结构，solver 本身无需知道是四重对称还是单轴对称。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the cavities of a mirrored frame by consulting shape-matched sprites.

    Input structure:
    - The dominant color is background.
    - One large monochrome frame encloses several holes; the hole pattern is only left-right symmetric, so the top and bottom can differ.
    - Outside the frame, small monochrome sprites repeat those hole shapes in arbitrary rotations or flips.

    Rule:
    - Locate the largest frame, crop to its bounding box, and begin from a solid block of the frame color.
    - Read each hole inside that box by shape, then gather every outside sprite with the same shape up to rotation/flip.
    - Fill the hole with the most frequent matching sprite color.

    Key insight:
    - The frame's symmetry only organizes where holes appear; the actual codebook is still shape -> plurality color among matching outside sprites.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected component of any non-bg color (8-connected).
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract hole components inside the bbox (anything not equal to frame color), 4-connected.
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    hole_mask = (sub != int(frame_col))
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = (mask.shape, mask.tobytes())
        holes.append((key, (int(hr0), int(hc0)), mask))

    # STEP 3) Extract sprites outside the bbox.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> color -> count
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        """Generate keys for all rotations and reflections of a boolean mask."""
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output: start from solid frame color, then fill each hole by shape match.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(out, np.full(hmask.shape, int(scol), dtype=int), int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (D8 variant) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 6
PLACEMENT_RETRIES = 80
GROUP_MULT_VALUES = [1, 2, 3]
GROUP_MULT_WEIGHTS = [4, 2, 1]
MIN_ROW_GAP = 4

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants (canonical representative)."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen, dedup = set(), []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _layout_metrics(occ: np.ndarray) -> dict[str, int | bool]:
    """Quantify whether the hole pattern visibly matches the LR-only variant."""
    hole_mask = ~occ
    row_counts = np.sum(hole_mask, axis=1).astype(int)
    mid = hole_mask.shape[0] // 2
    top_cells = int(np.sum(row_counts[:mid]))
    bottom_cells = int(np.sum(row_counts[(hole_mask.shape[0] + 1) // 2 :]))
    return {
        "lr_ok": bool(np.array_equal(hole_mask, np.fliplr(hole_mask))),
        "ud_ok": bool(np.array_equal(hole_mask, np.flipud(hole_mask))),
        "row_gap": int(np.sum(np.abs(row_counts - row_counts[::-1]))),
        "top_cells": top_cells,
        "bottom_cells": bottom_cells,
    }


def _candidate_rows(th: int, sh: int, band: str) -> list[int]:
    """Choose anchor rows from an upper band, lower band, or the whole interior."""
    max_r = th - sh - 1
    if max_r < 1:
        return []
    if band == "any":
        return list(range(1, max_r + 1))
    split = max(1, max_r // 2)
    if band == "top":
        return list(range(1, split + 1))
    if band == "bottom":
        start = min(max_r, split + 1)
        return list(range(start, max_r + 1))
    raise ValueError(f"unknown band: {band}")


def _mirror_lr_pair(r: int, c: int, sw: int, tw: int) -> list[tuple[int, int]]:
    """Return the left copy and its horizontal mirror."""
    c_m = tw - c - sw
    return [(r, c), (r, c_m)]


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, 4-disconnected from older holes, and no intra-group merge."""
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    rr, cc = pr + dr, pc + dc
                    if not occ[rr, cc]:
                        return None
                    new_cells.add((rr, cc))

    group_mask = np.zeros((th, tw), dtype=bool)
    for rr, cc in new_cells:
        group_mask[rr, cc] = True
    if len(components_mask(group_mask, connectivity=4)) != len(positions):
        return None

    for rr, cc in new_cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                if (nr, nc) not in new_cells:
                    return None
    return positions


def _try_place_lr_pair(occ, shape, th, tw, band):
    """Place one shape as a left-right mirrored pair, with row preference set by band."""
    sh, sw = shape.shape
    rows = _candidate_rows(th, sh, band)
    max_c = (tw - 2 * sw - 1) // 2
    if not rows or max_c < 1:
        return None
    r = random.choice(rows)
    c = random.randint(1, max_c)
    positions = _mirror_lr_pair(r, c, sw, tw)
    return _validate_positions(occ, shape, positions, th, tw)


def _place_shape_groups(occ, shape, th, tw, first_band, mult):
    """Attempt one or more mirrored placements for a shape. The first placement may be forced to a specific band."""
    bands = []
    if first_band is not None:
        bands.append(first_band)
    while len(bands) < mult:
        bands.append(random.choice(["any", "top", "bottom"]))

    placed = []
    for idx, band in enumerate(bands):
        positions = None
        for _ in range(PLACEMENT_RETRIES):
            positions = _try_place_lr_pair(occ, shape, th, tw, band)
            if positions is not None:
                break
        if positions is None:
            if idx == 0 and first_band is not None:
                return []
            continue
        _carve_hole(occ, shape, positions)
        placed.append(positions)
    return placed


def _carve_hole(occ, shape, positions):
    """Mark hole cells as unoccupied in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place holes with LR symmetry only. Returns (occ, hole_placements) or None."""
    max_sh = th - 2
    max_sw = max(1, (tw - 3) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    random.shuffle(available)
    available.sort(key=lambda shape: (shape.shape[1], int(np.sum(shape)), shape.shape[0]))
    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    used = set()

    # Force one upper-biased and one lower-biased shape so the output does not collapse back to UD symmetry.
    for forced_band in ("top", "bottom"):
        placed_this_band = False
        for idx, shape in enumerate(available):
            if idx in used:
                continue
            mult = random.choices(GROUP_MULT_VALUES, weights=GROUP_MULT_WEIGHTS)[0]
            placed = _place_shape_groups(occ, shape, th, tw, forced_band, mult)
            if not placed:
                continue
            hole_placements.append((shape, placed))
            used.add(idx)
            placed_this_band = True
            break
        if not placed_this_band:
            return None

    for idx, shape in enumerate(available):
        if idx in used or len(hole_placements) >= k_target:
            continue
        mult = random.choices(GROUP_MULT_VALUES, weights=GROUP_MULT_WEIGHTS)[0]
        placed = _place_shape_groups(occ, shape, th, tw, None, mult)
        if placed:
            hole_placements.append((shape, placed))
            used.add(idx)

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(positions) for _, groups in hole_placements for positions in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    metrics = _layout_metrics(occ)
    if not metrics["lr_ok"]:
        return None
    if metrics["ud_ok"]:
        return None
    if metrics["row_gap"] < MIN_ROW_GAP:
        return None
    if metrics["top_cells"] == 0 or metrics["bottom_cells"] == 0:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """Render the target output canvas and assign fill colors."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        patch = np.full(shape.shape, fill_col, dtype=int)
        for positions in groups:
            for pr, pc in positions:
                paste(output, patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    """Place a single sprite on inp and placement grids. Returns True on success."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """Place winning and losing sprites for each hole shape outside the frame."""
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        # Keep every losing color strictly below the winning count so the plurality is visually well defined.
        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build an LR-only frame, scatter sprites, validate."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc_pos = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc_pos)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
