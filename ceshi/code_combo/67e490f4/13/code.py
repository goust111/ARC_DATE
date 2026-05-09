# Combo variant of 67e490f4 (index 13)
# combo_spec: D7/1+D18/3
# Name: 大矩形洞+跨尺度同形

from __future__ import annotations

from collections import Counter
import math
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

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题: 框内有若干小型不规则洞, 外部精灵按精确形状(允许旋转/翻转)配对, 每种形状用外部同形精灵的多数颜色填充.
#    - D7/1 后: 洞改成大型矩形空腔, 外部块允许有缺角/缺口, 所以“精确 mask 匹配”失效, 只能看 bbox 尺寸.
#    - 再叠加 D18/3: 框内所有洞都属于同一个矩形比例族, 例如 3x4、6x8、9x12, 即同一基础矩形的不同整数缩放.
#      外部块也来自同一比例族, 但生成器故意让至少一种洞尺寸在外部没有同尺寸样本, 迫使求解器按最简长宽比
#      (例如 3:4) 进行跨尺度归类, 而不能只按精确尺寸配对.
#    - 具体例子: 若框内有 3x4、6x8 两种洞, 外部只出现多个 3x4 比例块(含 3x4 本体或 9x12 等同族缩放)并且主色为红,
#      那么 6x8 洞也必须填红; 旧版“按精确尺寸”逻辑会找不到 6x8 对应块.
# b) 复杂度校准
#    - 保留“最大框架 + 外部颜色多数投票 + 输出裁剪到框 bbox”的原题骨架.
#    - 洞仍然重复出现, 但只保留一个比例族和 2-3 个尺度, 这样跨尺度规律肉眼可见, 又不会因比例族过多而混乱.
#    - 外部加入 1-2 个非目标比例的干扰块族, 防止任务退化成“看所有外部块的总多数颜色”.
# c) 隐式假设审查
#    - `main()` 仍假设最大单色非背景连通块就是框架; 生成器让框面积远大于任一外部块.
#    - `main()` 现在假设框内每个洞都填满其 bbox, 即都是真矩形; 生成器只雕刻轴对齐矩形洞.
#    - `main()` 不再依赖洞与外部块 mask 完全一致, 只依赖 bbox 的最简比例键; 生成器让目标外部块都带缺口,
#      保证原题精确形状规则不会在这些样本上退化为同一行为.
#    - `main()` 假设所有洞共用同一个比例键且该比例键在外部有唯一多数颜色; 生成器显式保证这两点.
# d) 放置合法性
#    - 洞与框边界之间至少隔 1 个框色像素, 洞与洞之间也至少隔 1 格, 所以洞在框内始终 4-分离.
#    - 外部块用完整 bbox 作为 blocker 放置, 因而带缺口的块不会通过透明缺口彼此穿插.
# e) 视觉可观察性
#    - 同一输入里会同时看到多个不同尺寸但同长宽比的大矩形洞, 以及外部同长宽比的大块, 规律是直接可见的几何相似.
#    - 至少一种洞尺寸在外部缺少同尺寸样本, 所以“跨尺度同形”不是隐藏设定, 而是样例中真正需要用到的线索.
# f) 可视化整洁度
#    - 只保留一个大框架; 洞是规则矩形; 外部块彼此隔开; 干扰块比例有限, 整体仍保持 ARC 常见的清爽布局.
# g) 变化的可推断性
#    - 颜色规则仍是“看匹配族里的多数颜色”, 但匹配族从“同 mask”改成“同比例”.
#      颜色变化始终依附于清楚的几何分组, 不依赖颜色编号或不可见排序.
# h) 融合行为量化指标
#    - 关键指标 1: `missing_exact_size_holes` = 框内目标洞尺寸中, 外部目标比例块里没有同尺寸样本的尺寸个数.
#      该值必须 >= 1, 否则会退化回 D7/1 的“按精确尺寸”版本.
#    - 关键指标 2: `target_notched_fraction` = 目标比例外部块中非满矩形块的比例.
#      该值应接近 1, 否则会退化回原题的精确 mask 版本.


def _ratio_key(h: int, w: int) -> tuple[int, int]:
    g = math.gcd(int(h), int(w))
    a = int(h) // g
    b = int(w) // g
    return (a, b) if a <= b else (b, a)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = tuple(map(int, frame["bbox"]))
    r0, c0, r1, c1 = fb

    sub = crop_bbox(g, fb)
    th, tw = map(int, sub.shape)
    hole_mask = sub != int(frame_col)

    holes: list[dict] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        assert bool(mask.all())
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "size": (int(hh), int(ww)),
                "ratio": _ratio_key(hh, ww),
            }
        )

    assert holes
    hole_ratios = {hole["ratio"] for hole in holes}
    assert len(hole_ratios) == 1
    target_ratio = next(iter(hole_ratios))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_ratio: dict[tuple[int, int], dict[int, int]] = {}
    sprite_size_counts: dict[tuple[int, int], Counter] = {}
    sprite_notched: dict[tuple[int, int], list[bool]] = {}
    for comp in spr_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        ratio = _ratio_key(hh, ww)
        vals = [int(v) for v in g[comp["coords"][:, 0], comp["coords"][:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        area = int(comp["coords"].shape[0])

        bucket = sprites_by_ratio.setdefault(ratio, {})
        bucket[scol] = int(bucket.get(scol, 0)) + 1
        sprite_size_counts.setdefault(ratio, Counter())[(hh, ww)] += 1
        sprite_notched.setdefault(ratio, []).append(area < hh * ww)

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "frame_bbox": fb,
        "frame_shape": (int(th), int(tw)),
        "holes": holes,
        "target_ratio": target_ratio,
        "sprites_by_ratio": sprites_by_ratio,
        "sprite_size_counts": sprite_size_counts,
        "sprite_notched": sprite_notched,
    }


def describe_instance(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    target_ratio = scene["target_ratio"]
    hole_sizes = sorted({tuple(map(int, hole["size"])) for hole in scene["holes"]})
    target_sprite_sizes = set(scene["sprite_size_counts"].get(target_ratio, Counter()).keys())
    missing_exact = sum(1 for size in hole_sizes if size not in target_sprite_sizes)
    notched_flags = scene["sprite_notched"].get(target_ratio, [])
    target_notched_fraction = 0.0 if not notched_flags else sum(notched_flags) / len(notched_flags)
    return {
        "target_ratio": tuple(map(int, target_ratio)),
        "hole_sizes": hole_sizes,
        "hole_size_count": len(hole_sizes),
        "missing_exact_size_holes": int(missing_exact),
        "target_sprite_sizes": sorted(target_sprite_sizes),
        "target_notched_fraction": float(target_notched_fraction),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill large rectangular cavities inside a frame using outside chunks that match by scaled rectangle family.

    Input structure:
    - bg = most common color. One large monochrome frame encloses several big rectangular cavities.
    - Every cavity is a full axis-aligned rectangle, and all cavities share one aspect ratio but appear at different integer scales.
    - Outside the frame sit large monochrome chunks with the same ratio family or distractor ratios; target chunks may have visible bites.

    Rule:
    - Crop to the frame bbox and start from a solid frame-colored rectangle.
    - Read each cavity by its bbox size, reduce height/width to a canonical aspect-ratio key, and verify all cavities belong to one family.
    - Among outside chunks with that same ratio key, choose the unique plurality color and use it to fill every cavity, regardless of the cavity's exact scale.

    Key insight:
    - The external pieces are not exact stencils and not even required to share the same absolute size as a hole: they are scale-coded ratio witnesses, so a 3x4-style chunk can color a 6x8 cavity if both reduce to the same rectangle proportion.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    th, tw = map(int, scene["frame_shape"])
    frame_col = int(scene["frame_col"])
    target_ratio = scene["target_ratio"]
    cmap = scene["sprites_by_ratio"].get(target_ratio)
    assert cmap is not None
    best = max(int(v) for v in cmap.values())
    winners = [int(col) for col, cnt in cmap.items() if int(cnt) == best]
    assert len(winners) == 1
    fill_col = int(winners[0])

    out = np.full((th, tw), int(frame_col), dtype=int)
    for hole in scene["holes"]:
        hr0, hc0, hr1, hc1 = map(int, hole["bbox"])
        out[hr0 : hr1 + 1, hc0 : hc1 + 1] = int(fill_col)
    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (18, 22)
FRAME_W_RANGE = (18, 24)

BASE_RATIO_CHOICES = [(3, 3), (3, 4)]
FRAME_DENSITY_RANGE = (0.48, 0.80)
PLACEMENT_RETRIES = 220
MAX_GEN_ATTEMPTS = 500

TARGET_WINNING_RANGE = (4, 6)
TARGET_LOSER_COLOR_RANGE = (1, 2)
TARGET_LOSER_TOTAL_MAX = 3
DISTRACTOR_FAMILY_RANGE = (1, 2)
DISTRACTOR_PER_FAMILY_RANGE = (2, 4)
SPRITE_PADDING = 1


def _hole_scale_plan(base_h: int, base_w: int, th: int, tw: int) -> list[int] | None:
    max_scale = min((th - 2) // base_h, (tw - 2) // base_w)
    if max_scale < 2:
        return None
    options: list[list[int]] = [[1, 2]]
    if (base_h, base_w) == (3, 3) and max_scale >= 3:
        options.append([1, 2, 3])
    return random.choice(options)


def _hole_multiplicities(scales: list[int]) -> dict[int, int]:
    if scales == [1, 2, 3]:
        return {1: random.randint(2, 3), 2: random.randint(1, 2), 3: 1}
    return {1: random.randint(3, 4), 2: random.randint(2, 3)}


def _can_place_rect(occ: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    th, tw = map(int, occ.shape)
    if r < 1 or c < 1 or r + h > th - 1 or c + w > tw - 1:
        return False
    return bool(occ[r - 1 : r + h + 1, c - 1 : c + w + 1].all())


def _build_hole_layout(
    th: int, tw: int
) -> tuple[np.ndarray, list[dict], tuple[int, int], list[int]] | None:
    family_choices = BASE_RATIO_CHOICES[:]
    random.shuffle(family_choices)
    for base_h, base_w in family_choices:
        scales = _hole_scale_plan(base_h, base_w, th, tw)
        if scales is None:
            continue

        occ = np.ones((th, tw), dtype=bool)
        multiplicities = _hole_multiplicities(scales)
        rects: list[tuple[int, int, int]] = []
        for scale in scales:
            for _ in range(multiplicities[scale]):
                rects.append((base_h * scale, base_w * scale, scale))
        rects.sort(key=lambda item: -(item[0] * item[1]))

        holes: list[dict] = []
        failed = False
        for h, w, scale in rects:
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                r = random.randint(1, th - h - 1)
                c = random.randint(1, tw - w - 1)
                if not _can_place_rect(occ, r, c, h, w):
                    continue
                occ[r : r + h, c : c + w] = False
                holes.append({"r": int(r), "c": int(c), "h": int(h), "w": int(w), "scale": int(scale)})
                placed = True
                break
            if not placed:
                failed = True
                break

        if failed:
            continue

        density = 1.0 - int(np.sum(~occ)) / float(th * tw)
        if not (FRAME_DENSITY_RANGE[0] <= density <= FRAME_DENSITY_RANGE[1]):
            continue
        return occ, holes, (int(base_h), int(base_w)), [int(s) for s in scales]
    return None


def _choose_frame_origin(th: int, tw: int) -> tuple[int, int]:
    corners = [
        (0, 0),
        (0, GRID_SIZE - tw),
        (GRID_SIZE - th, 0),
        (GRID_SIZE - th, GRID_SIZE - tw),
    ]
    if random.random() < 0.85:
        return random.choice(corners)
    return random.randint(0, GRID_SIZE - th), random.randint(0, GRID_SIZE - tw)


def _intended_frame_is_dominant(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_col: int
) -> bool:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    if not comps:
        return False
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    if len(frame_candidates) != 1:
        return False
    frame = frame_candidates[0]
    return int(frame["color"]) == int(frame_col) and tuple(map(int, frame["bbox"])) == tuple(map(int, frame_bbox))


def _build_output(
    holes: list[dict], th: int, tw: int, frame_col: int, fill_col: int
) -> np.ndarray:
    out = np.full((th, tw), int(frame_col), dtype=int)
    for hole in holes:
        r = int(hole["r"])
        c = int(hole["c"])
        h = int(hole["h"])
        w = int(hole["w"])
        out[r : r + h, c : c + w] = int(fill_col)
    return out


def _sprite_variants(h: int, w: int) -> list[np.ndarray]:
    full = np.ones((h, w), dtype=bool)
    variants: list[np.ndarray] = []
    remove_sets = [
        [(0, 0)],
        [(0, w - 1)],
        [(h - 1, 0)],
        [(h - 1, w - 1)],
    ]
    if w >= 4:
        remove_sets.extend([[(0, 0), (0, 1)], [(h - 1, w - 2), (h - 1, w - 1)]])
    if h >= 4:
        remove_sets.extend([[(0, 0), (1, 0)], [(h - 2, w - 1), (h - 1, w - 1)]])
    if h >= 3 and w >= 3:
        remove_sets.append([(h // 2, w // 2)])

    seen: set[bytes] = set()
    for cells in remove_sets:
        mask = full.copy()
        for rr, cc in cells:
            mask[rr, cc] = False
        if not bool(mask.any()) or bool(mask.all()):
            continue
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        variants.append(mask)
    assert variants
    return variants


def _place_sprite(inp: np.ndarray, placement: np.ndarray, mask: np.ndarray, color: int, bg: int) -> bool:
    h, w = map(int, mask.shape)
    blocker = np.full((h, w), int(color), dtype=int)
    patch = np.full((h, w), int(bg), dtype=int)
    patch[mask] = int(color)
    try:
        r, c = random_free_location(placement, blocker, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, blocker, int(r), int(c))
    paste(inp, patch, int(r), int(c), transparent=bg)
    return True


def _target_sprite_scales(hole_scales: list[int]) -> list[int]:
    if len(hole_scales) >= 3:
        missing = random.choice(hole_scales)
        kept = [s for s in hole_scales if s != missing]
        assert len(kept) >= 2
        return kept
    return [random.choice(hole_scales)]


def _scatter_target_sprites(
    inp: np.ndarray,
    placement: np.ndarray,
    base_shape: tuple[int, int],
    hole_scales: list[int],
    fill_col: int,
    bg: int,
    frame_col: int,
) -> tuple[bool, list[int]]:
    base_h, base_w = map(int, base_shape)
    sprite_scales = _target_sprite_scales(hole_scales)
    winning_total = random.randint(*TARGET_WINNING_RANGE)

    scale_pool = [random.choice(sprite_scales) for _ in range(winning_total)]
    if len(sprite_scales) >= 2:
        for scale in sprite_scales:
            if scale not in scale_pool:
                scale_pool[random.randrange(len(scale_pool))] = int(scale)

    for scale in scale_pool:
        h = base_h * int(scale)
        w = base_w * int(scale)
        variants = _sprite_variants(h, w)
        if not _place_sprite(inp, placement, random.choice(variants), fill_col, bg):
            return False, sprite_scales

    other_pool = [c for c in range(NUM_COLORS) if c not in {bg, frame_col, fill_col}]
    if not other_pool:
        return False, sprite_scales
    loser_color_count = random.randint(TARGET_LOSER_COLOR_RANGE[0], min(TARGET_LOSER_COLOR_RANGE[1], len(other_pool)))
    loser_cols = random.sample(other_pool, loser_color_count)
    max_total = min(TARGET_LOSER_TOTAL_MAX, max(1, winning_total - 1))
    losing_budget = random.randint(1, max_total)
    for _ in range(losing_budget):
        scale = random.choice(sprite_scales)
        h = base_h * int(scale)
        w = base_w * int(scale)
        variants = _sprite_variants(h, w)
        color = random.choice(loser_cols)
        if not _place_sprite(inp, placement, random.choice(variants), color, bg):
            break
    return True, sprite_scales


def _distractor_candidates(target_ratio: tuple[int, int]) -> list[tuple[int, int]]:
    candidates = [(3, 5), (4, 5), (4, 7), (5, 8), (6, 5)]
    return [size for size in candidates if _ratio_key(*size) != tuple(map(int, target_ratio))]


def _scatter_distractors(
    inp: np.ndarray,
    placement: np.ndarray,
    target_ratio: tuple[int, int],
    bg: int,
    frame_col: int,
    fill_col: int,
) -> bool:
    families = _distractor_candidates(target_ratio)
    if not families:
        return True
    family_count = random.randint(DISTRACTOR_FAMILY_RANGE[0], min(DISTRACTOR_FAMILY_RANGE[1], len(families)))
    chosen = random.sample(families, family_count)
    color_pool = [c for c in range(NUM_COLORS) if c not in {bg, frame_col, fill_col}]
    if len(color_pool) < family_count:
        return False
    family_colors = random.sample(color_pool, family_count)

    for (h, w), color in zip(chosen, family_colors):
        variants = _sprite_variants(h, w)
        count = random.randint(*DISTRACTOR_PER_FAMILY_RANGE)
        placed_any = False
        for _ in range(count):
            if _place_sprite(inp, placement, random.choice(variants), color, bg):
                placed_any = True
            else:
                break
        if not placed_any:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, holes, base_shape, hole_scales = layout

    fill_col = random.choice([c for c in range(NUM_COLORS) if c not in {bg, frame_col}])
    output = _build_output(holes, th, tw, frame_col, fill_col)

    fr, fc = _choose_frame_origin(th, tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    frame_patch = np.where(occ, int(frame_col), int(bg)).astype(int)
    paste(inp, frame_patch, int(fr), int(fc))

    placement = inp.copy()
    placement[fr : fr + th, fc : fc + tw] = int(frame_col)
    ok, sprite_scales = _scatter_target_sprites(inp, placement, base_shape, hole_scales, fill_col, bg, frame_col)
    if not ok:
        return None
    if not _scatter_distractors(inp, placement, _ratio_key(*base_shape), bg, frame_col, fill_col):
        return None

    expected_bbox = (int(fr), int(fc), int(fr + th - 1), int(fc + tw - 1))
    if not _intended_frame_is_dominant(inp, expected_bbox, frame_col):
        return None

    metrics = describe_instance(inp)
    if metrics["hole_size_count"] < 2:
        return None
    if metrics["missing_exact_size_holes"] < 1:
        return None
    if metrics["target_notched_fraction"] < 0.99:
        return None
    if sorted(metrics["target_ratio"]) != sorted(_ratio_key(*base_shape)):
        return None
    if set(metrics["target_sprite_sizes"]) == set((base_shape[0] * s, base_shape[1] * s) for s in hole_scales):
        return None
    if not set(sprite_scales).issubset(set(hole_scales)):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
