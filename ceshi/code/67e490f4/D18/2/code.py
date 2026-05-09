# Variant of 67e490f4
# Dimension: D18 — 洞形状库的多样性 (meta)
# Change: 使用预定义的形状库（矩形、L形、T形、十字形等 16 种基础形状），3-6 种被选入每道题... → 随机生成洞形状（非预定义库）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the D18 variant
# ----------------------------------
# spec_like_analysis:
#   concrete_behavior:
#     - 输入仍然是一个大框架 + 框外散落的小精灵；输出仍然裁剪到框架 bbox。
#     - 变化只在“洞形状的来源”：原题从固定 16 形状库里挑 3-6 种；这里改成每题现场随机长出 3-6 个 4 连通布尔掩码。
#     - 洞与框外精灵仍然通过“形状在旋转/翻转下相同”来配对；颜色规则仍然是匹配精灵里出现次数最多的颜色。
#   behavior_trace:
#     - 原逻辑示例：洞是一个 2x2 方块，框外也散落多个 2x2 方块精灵，solver 直接按 mask 字节串匹配。
#     - 变体逻辑示例：洞可能是一个 4x4 bbox 内面积 6 的不规则连通掩码，例如锯齿形；框外给出它的若干旋转/翻转版本。
#     - 差异不在 solver 的配对步骤，而在 generator 不再枚举有限类别，而是随机生成、裁剪、去重并验证连通性。
#   complexity_calibration:
#     - 若随机形状太简单，视觉上会退化回原题，所以这里避免单行/单列/满矩形，并额外拒绝落入原始 16 形状库的 canonical key。
#     - 若自由度过高，易出现不可放置或过密布局，所以仍保留：bbox <= 5x6、面积 <= 8、3-6 种形状、对称散布、洞之间 4 分离。
#   main_line_assumptions:
#     - 找最大单色连通块当框架：要求框架面积远大于任一外部精灵；generator 保持这一点。
#     - 在框架 bbox 内把 != frame_col 的区域当洞：要求洞不接触 bbox 边界，且 bbox 内不引入别的杂色；generator 保持边框完整。
#     - 洞 key 是“当前朝向”的精确 mask，精灵 key 枚举所有旋转/翻转：这要求每个洞至少有一个同 canonical shape 的外部精灵；generator 显式构造。
#     - 若多个匹配颜色并列，solver 仍按原代码保留一个稳定 tie-break；但 generator 会避免制造并列，因此人类无需依赖颜色 id 推理。
#   placement_legality:
#     - 随机形状先用 4 邻域生长保证连通，再裁剪到其真实 bbox。
#     - 放置时沿用原任务的占用网格和 4 邻域隔离检查，保证洞不重叠、不越界、不贴边、不与其他洞 4 接触。
#     - 精灵仍使用 random_free_location 且要求 Chebyshev padding=1，确保与框和其他精灵留出背景间隔。
#   visual_observability:
#     - 关键可见变化是：每题出现的洞轮廓不再像固定库里的“方块/L/T/十字”，而是每题新生成的不规则轮廓。
#     - 为防止样例偶然长得像旧库，生成器把原始 16 形状库仅作为 blacklist 使用，不参与采样，只用于拒绝过旧的轮廓。
#   cleanliness:
#     - 仍维持单一大框架、洞留在框内、精灵留在框外的分区布局，不叠加额外标记层。
#   color_inference:
#     - 颜色机制未变，但生成器保证每个 shape 的 winning color 严格多数，避免需要通过颜色 id 打破肉眼不可见的平局。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped filled frame by matching arbitrary hole silhouettes to outside sprites.

    Input structure:
    - The background is the most common color, with one large monochrome frame standing out.
    - Inside that frame bbox are several disconnected holes, each an arbitrary connected mask rather than a fixed library piece.
    - Outside the frame bbox are small monochrome sprites whose silhouettes match those holes up to rotation or flip.

    Rule:
    - Locate the large frame, crop to its bounding box, and start from a solid rectangle of the frame color.
    - Extract each hole mask exactly as it appears, then gather every outside sprite under all rotated/flipped variants.
    - Fill each hole with the color that is most frequent among sprites sharing that shape, and return the cropped result.

    Key insight:
    - The solver does not need a known catalogue of shapes: the outside sprites themselves define the per-puzzle shape dictionary, and only rotation/flip-invariant shape matching matters.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # The D18 change does not affect frame detection: the frame is still the largest
    # monochrome non-background component in the scene.
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

    # Hole extraction is shape-agnostic: any 4-connected non-frame region inside the
    # bbox becomes a target mask, regardless of whether it resembles a known primitive.
    hole_mask = (sub != int(frame_col))
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # Remove the frame area so only outside sprites remain for shape/color voting.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
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
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MIN_AREA = 4
SHAPE_MAX_AREA = 8
SHAPE_MAX_H = 5
SHAPE_MAX_W = 6
FRAME_DENSITY_RANGE = (0.55, 0.90)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])
RANDOM_SHAPE_ATTEMPTS = 120
LAYOUT_SHAPE_ATTEMPTS = 120

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _baseline_library_keys():
    # Blacklist only: this preserves the D18 change by rejecting silhouettes that
    # would visually fall back to the original fixed shape vocabulary.
    raw = [
        [[1]], [[1, 1]], [[1], [1]], [[1, 1, 1]], [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]], [[1, 0], [1, 1]], [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]], [[1, 1], [1, 0]], [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]],
    ]
    return {_canonical_key(np.array(shape, dtype=bool)) for shape in raw}


BASELINE_LIBRARY_KEYS = _baseline_library_keys()


def _mask_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _grow_connected_shape(max_h: int, max_w: int) -> np.ndarray | None:
    h = random.randint(2, max_h)
    w = random.randint(2, max_w)
    if h * w <= SHAPE_MIN_AREA:
        return None

    area_values = [a for a in range(SHAPE_MIN_AREA, min(SHAPE_MAX_AREA, h * w - 1) + 1)]
    if not area_values:
        return None
    area_weights = [SHAPE_MAX_AREA - a + 2 for a in area_values]
    target_area = random.choices(area_values, weights=area_weights, k=1)[0]

    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    while len(cells) < target_area:
        frontier = []
        seen = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    return _mask_from_cells(cells, h, w)


def _is_visibly_novel(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(np.sum(mask))
    if h < 2 or w < 2:
        return False
    if area <= 0 or area == h * w:
        return False
    if np.any(mask.sum(axis=0) == area) or np.any(mask.sum(axis=1) == area):
        return False
    return _canonical_key(mask) not in BASELINE_LIBRARY_KEYS


def _random_shape(max_h: int, max_w: int, used_keys: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(RANDOM_SHAPE_ATTEMPTS):
        mask = _grow_connected_shape(max_h, max_w)
        if mask is None:
            continue
        area = int(np.sum(mask))
        h, w = mask.shape
        if area < SHAPE_MIN_AREA or area > SHAPE_MAX_AREA:
            continue
        if h > max_h or w > max_w:
            continue
        if not _is_visibly_novel(mask):
            continue
        key = _canonical_key(mask)
        if key in used_keys:
            continue
        return mask
    return None


def _mirror_pos(r, c, sh, sw, th, tw):
    return list({
        (r, c),
        (r, tw - 1 - c - sw + 1),
        (th - 1 - r - sh + 1, c),
        (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
    })


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_symmetric(occ, shape, th, tw):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    max_sh = min(SHAPE_MAX_H, max(2, (th - 2) // 2))
    max_sw = min(SHAPE_MAX_W, max(2, (tw - 2) // 2))
    if max_sh < 2 or max_sw < 2:
        return None

    k_target = random.randint(*SHAPE_COUNT_RANGE)
    even_dims = (th % 2 == 0 or tw % 2 == 0)

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()

    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS
    for _ in range(LAYOUT_SHAPE_ATTEMPTS):
        if len(hole_placements) >= k_target:
            break
        shape = _random_shape(max_sh, max_sw, used_keys)
        if shape is None:
            continue
        mult = random.choices(mult_vals, weights=mult_wts, k=1)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    break
        if placed:
            used_keys.add(_canonical_key(shape))
            hole_placements.append((shape, placed))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(positions) for _, groups in hole_placements for positions in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in groups:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
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

        for oc in oc_list:
            max_loser = max(1, n_winning - 1)
            n_oc = random.randint(1, max_loser)
            if n_oc >= n_winning:
                n_oc = n_winning - 1
            if n_oc <= 0:
                continue
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
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
