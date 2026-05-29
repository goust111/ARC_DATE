# Combo variant of c7f57c3e (index 11)
# combo_spec: D1/1+D9/3
# Name: 按大小分组 + 条件性交换
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    这个融合版保留原题的两种多色原型、共享的 color-1 锚点、整数倍缩放与分散放置，但把“大小”拆成两套不同规则：
#    1. 分组规则：按 bbox 面积把精灵分成 small group 和 large group。
#       这里固定出现 3 档面积：最小、居中、最大；其中 small group = {最小, 居中}，large group = {最大}。
#    2. 参与规则：只有 bbox 面积严格大于最小面积的精灵才交换；最小面积精灵保持不变。
#    一个具体例子：
#      - 输入含 4 个精灵：1x 原型 A、2x 原型 B、3x 原型 A、3x 原型 A。
#      - 依面积分组后，1x 和 2x 属于 small group，两个 3x 属于 large group。
#      - 但按参与阈值判断时，只有 2x 与两个 3x 交换，1x 不交换。
#      - 因而输出变成：1x 仍是 A，2x 变成 A，两个 3x 变成 B。
#    差异点在于：原题是“识别原型后全部切换”；本融合版必须先做大小分组，再单独判断是否越过最小面积阈值，
#    所以中等尺寸精灵会出现“属于 small group 但仍然参与交换”的现象。
#
# b) 复杂度校准
#    只保留 3 档尺寸与 2 个组，避免引入太多自由度。4 个对象里：
#      - 最小尺寸恰好 1 个，作为不参与交换的锚定示例；
#      - 中等尺寸恰好 1 个，负责体现“组别与参与条件不同”；
#      - 最大尺寸恰好 2 个，给 large group 提供重复支撑。
#    这样既不会退化成“所有大精灵都换”，也不会因为尺寸档位太多而让样例难读。
#
# c) 隐式假设审查
#    原 solver 默认“所有对象都要切到另一原型”，这里不再成立，最小面积对象必须跳过改写。
#    原 D1 单变体中“同一大小组里所有输入对象都 already 显示同一原型”也不再成立：
#      - small group 内部会同时出现一个未交换的最小精灵和一个待交换的中等精灵。
#    因此新 solver 不能从“某组任意成员”直接读出该组的目标原型。
#    这里改成：
#      - 最小面积精灵直接给出 small group 的目标原型；
#      - 中等面积精灵在输入中显示的是 opposite prototype，经下采样后恢复出 large group 的目标原型；
#      - 最大面积精灵都应是 small-group prototype 的放大副本，因为它们在输入端是已交换状态。
#    这些前提都由 generator 固定构造并用 oracle 校验。
#
# d) 放置合法性
#    仍采用 output-first。先决定每个位置在输出里应该显示哪一种原型，再反推出输入是否需要预先交换。
#    放置时继续对“输出 patch / 输入 patch 的锚点对齐并集 bbox”预留矩形，保证输入和输出都不越界、不重叠。
#
# e) 视觉可观察性
#    人眼能直接看到三档尺寸：最小、中等、最大。对照输入输出时，会看到：
#      - 最小精灵不变；
#      - 中等精灵会变，但它仍和最小精灵归在同一组；
#      - 最大精灵全部变成另一种颜色模式。
#    规则完全基于可见的尺寸层级，不依赖颜色数值大小或隐式排序。
#
# f) 可视化整洁度
#    总对象数固定为 4，且通过 reserve + padding 保证留白。最大尺度只放 2 个，其余为 1 个中等和 1 个最小，
#    版面不会过于拥挤，组别与变化对象都容易逐个对应。
#
# g) 变化的可推断性
#    颜色变化仍限定在两种结构相同但配色不同的原型之间，color-1 锚点始终不变。
#    由于中等精灵在输入中与最小精灵颜色模式不同、但输出里会与最小精灵对齐，解题者能观察到：
#    “是否交换”不是简单按组别决定，而是还要看是否超过最小面积阈值。
#
# h) 新的 main() 隐式假设与 generator 保证
#    `main()` 新假设：
#      - 恰好有 4 个连通精灵；
#      - bbox 面积恰好有 3 档，计数分布为 1 / 1 / 2；
#      - 最小与中等精灵都属于 small group，最大精灵属于 large group；
#      - 最小精灵给出 small-group prototype；
#      - 中等精灵是 large-group prototype 的整数倍缩放；
#      - 最大精灵都是 small-group prototype 的更大倍数缩放；
#      - 两种 prototype 共享同一 bbox 形状和 color-1 锚点尺寸。
#    `generate_input()` 会强制使用 1x / 2x / 3x / 3x 的规模配置，并用 `np.array_equal(main(inp), out)` 做最终校验。

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.append(extra_s)

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array must contain color-1 anchor pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _downscale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "sprite shape must be divisible by scale"
    return arr[::scale, ::scale].copy()


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _infer_scale(arr: np.ndarray, base_shape: tuple[int, int]) -> int:
    base_h, base_w = base_shape
    h, w = arr.shape
    assert h % base_h == 0 and w % base_w == 0, "sprite must be an integer upscale of the base shape"
    scale_h = h // base_h
    scale_w = w // base_w
    assert scale_h == scale_w, "height/width scale must agree"
    return int(scale_h)


def _paint_replacement(
    out: np.ndarray,
    pts: np.ndarray,
    bbox: tuple[int, int, int, int],
    arr: np.ndarray,
    target_proto: np.ndarray,
    scale: int,
    bg: int,
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = bbox
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, _ar1, _ac1 = anc

    for rr, cc in pts:
        out[int(rr), int(cc)] = bg

    scaled_target = _scale_nn(target_proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(target_proto, bg)
    top = (r0 + ar0) - (pr0 * scale)
    left = (c0 + ac0) - (pc0 * scale)

    ph, pw = scaled_target.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    sub = scaled_target[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    mask = sub != bg
    out_slice = out[rr0:rr1, cc0:cc1]
    out_slice[mask] = sub[mask]
    out[rr0:rr1, cc0:cc1] = out_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange group-level color patterns by size class, but only for sprites above the minimum-area threshold.

    Input structure:
    - The grid contains exactly four separated multicolor sprites built from two anchor-compatible prototypes.
    - Their bbox areas occur in three levels: one smallest sprite, one medium sprite, and two largest sprites.
    - The smallest and medium sprites form the small-size group; the two largest sprites form the large-size group.

    Rule:
    - Use bbox area to split sprites into the two size groups, but use a stricter condition for participation: only sprites with area greater than the minimum area are allowed to change.
    - The minimum-area sprite stays fixed and reveals the small-group prototype.
    - The medium sprite switches to that small-group prototype, while the largest sprites switch to the other prototype, always preserving each sprite's own scale and anchor alignment.

    Key insight: Size drives two different decisions. Relative size class says which group a sprite belongs to, but the absolute minimum-area cutoff says whether that group member is frozen or swappable, so the medium sprite is the crucial case where those two rules diverge.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 4, "combo variant expects exactly 4 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [_bbox_area(s["bbox"]) for s in sprites]
    area_levels = sorted(set(areas))
    assert len(area_levels) == 3, "combo variant expects exactly 3 bbox-area levels"
    min_area, mid_area, max_area = area_levels
    assert areas.count(min_area) == 1, "must have exactly one minimum-area sprite"
    assert areas.count(mid_area) == 1, "must have exactly one medium-area sprite"
    assert areas.count(max_area) == 2, "must have exactly two maximum-area sprites"

    min_sprite = next(s for s, area in zip(sprites, areas, strict=False) if area == min_area)
    mid_sprite = next(s for s, area in zip(sprites, areas, strict=False) if area == mid_area)
    max_sprites = [s for s, area in zip(sprites, areas, strict=False) if area == max_area]

    small_proto = min_sprite["arr"].copy()
    base_shape = small_proto.shape

    mid_scale = _infer_scale(mid_sprite["arr"], base_shape)
    max_scale = _infer_scale(max_sprites[0]["arr"], base_shape)
    assert mid_scale >= 2, "medium sprite must be enlarged relative to the minimum sprite"
    assert max_scale > mid_scale, "largest sprites must use a larger scale than the medium sprite"

    large_proto = _downscale_nn(mid_sprite["arr"], mid_scale)
    assert not np.array_equal(small_proto, large_proto), "the two recovered prototypes must differ"

    small_anchor = _pick_anchor_bbox(small_proto, bg)
    large_anchor = _pick_anchor_bbox(large_proto, bg)
    small_anchor_hw = (small_anchor[2] - small_anchor[0] + 1, small_anchor[3] - small_anchor[1] + 1)
    large_anchor_hw = (large_anchor[2] - large_anchor[0] + 1, large_anchor[3] - large_anchor[1] + 1)
    assert small_anchor_hw == large_anchor_hw, "prototype anchors must share the same size"

    assert np.array_equal(_downscale_nn(mid_sprite["arr"], mid_scale), large_proto), "medium sprite should normalize to the large-group prototype"
    for s in max_sprites:
        scale = _infer_scale(s["arr"], base_shape)
        assert scale == max_scale, "largest sprites must share one common scale"
        norm = _downscale_nn(s["arr"], scale)
        assert np.array_equal(norm, small_proto), "largest input sprites should normalize to the small-group prototype"

    out = g.copy()

    _paint_replacement(
        out,
        mid_sprite["pts"],
        mid_sprite["bbox"],
        mid_sprite["arr"],
        small_proto,
        mid_scale,
        bg,
    )
    for s in max_sprites:
        _paint_replacement(
            out,
            s["pts"],
            s["bbox"],
            s["arr"],
            large_proto,
            max_scale,
            bg,
        )

    return out


# --------------- generator (o5x combo variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 24
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.0

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SMALL_SCALE = 1
MID_SCALE = 2
LARGE_SCALE = 3
LARGE_COUNT = 2

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: VARIABLE_ROW_WEIGHTS,
        3: VARIABLE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        entries = weights_by_role[role]
        kinds, wts = zip(*entries)
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        masks[i] = _row_mask("outer" if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB else "full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, a_cols), _colorize(np.flipud(labels), b_cols)


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _build_instances() -> list[dict[str, int | bool]]:
    small_out_type = random.randrange(2)
    large_out_type = 1 - small_out_type
    instances = [
        {"scale": SMALL_SCALE, "out_type": small_out_type, "will_swap": False},
        {"scale": MID_SCALE, "out_type": small_out_type, "will_swap": True},
    ]
    for _ in range(LARGE_COUNT):
        instances.append({"scale": LARGE_SCALE, "out_type": large_out_type, "will_swap": True})
    random.shuffle(instances)
    instances.sort(key=lambda item: int(item["scale"]), reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]
    instances = _build_instances()

    max_dim = max(max(protos[int(item["out_type"])].shape) * int(item["scale"]) for item in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for item in instances:
        out_type = int(item["out_type"])
        scale = int(item["scale"])
        will_swap = bool(item["will_swap"])
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[1 - out_type] if will_swap else protos[out_type], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
