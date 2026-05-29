# Combo variant of c7f57c3e (index 28)
# combo_spec: D6/3+D12/0
# Name: 颜色取反 + 按像素级排列定义模式
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题：每个 sprite 属于两类 prototype 之一；输出时整块切到“另一类 prototype”的整套颜色模式。
#    - 融合后：sprite 仍然只有两类 prototype，但这两类故意使用同一对输入颜色 {2,3}，只靠像素级排列区分：
#      例如 A 是“上方较多 red(2)、下方较多 green(3)”的具体布置，B 则是它的上下翻转；两者颜色集合相同，
#      但逐像素排布不同，所以 D12/0 的区分点是“形状内部哪一格是什么颜色”，不是“用了哪两种颜色”。
#    - 对任意一个 A 或 B 实例：原题会换成另一类 prototype；融合后不再跨类交换，而是在原地对每个非 bg、
#      非 1 像素做互补映射 2->8、3->7（并保留 1 作为稳定骨架色）。因此位置、尺度、support 完全不变，只改颜色。
#    - 小例子：若 1x A 原型中某一列从上到下是 [2,1,3]，原题会改成 B 的对应像素排列；融合后仍保持这一列的
#      空间顺序，只把它变成 [8,1,7]。因此“哪一格原本是 2/3”直接决定输出里哪一格变成 8/7。
# b) 复杂度校准
#    - 只保留两个精确 prototype、3-4 个实例、且仅一个大尺度实例，离散情况仍然很少，不会让规律过散。
#    - 两个 prototype 共用同一输入颜色集，强制人类不能只按颜色集合分组；同时它们又是明显不同的上下排布，
#      2-3 个样例就能看出“同色集里其实有两种像素级模式”。
#    - 互补映射只使用两对稳定颜色：输入侧 {2,3}，输出侧 {8,7}。颜色变化少而结构清楚，避免无提示的大规模重映射。
# c) 隐式假设审查（对原 main() 的逐段影响）
#    - 原 `components_mask(g != bg)` 仍然成立：生成器继续保证每个 sprite 是单个 8 连通组件。
#    - 原 “最小面积 sprite 恰好给出 2 个 prototype” 作为数据集结构仍保留，但 solver 不再依赖它做实际变换；
#      因为 D6/3 把跨 sprite 的配对关系删除了，输出只取决于本 sprite 内部像素颜色。
#    - 原 `best_idx / repl_idx = 1 - best_idx` 的类别切换前提已失效：融合后不存在“换到对方 prototype”这一步。
#    - 原基于 anchor bbox 的尺度推断、清空旧 patch、再按 anchor 放置新 patch，这些都不再必要；因为新规则
#      保持 support 不动，最大重叠就是逐像素原地改色。
#    - 新 solver 的前提变成：有效 sprite 里的可变颜色必须来自已知互补对；generate_input() 只使用 {2,3}，
#      因而这一前提总成立。
# d) 放置合法性
#    - 生成器直接放置输入 patch；由于输出与输入 footprint 完全相同，不存在 swap 后越界或与别的 sprite 重新碰撞的问题。
#    - 仍使用 `random_free_location` + padding>=2 + border>=1，确保实例彼此不重叠、不贴边，视觉上干净。
# e) 视觉可观察性
#    - 人类能直接看到两类对象用了同样的两种输入色，但上下/局部排布不同，这就是 D12/0 的可见线索。
#    - 输出中所有 2 都统一变 8、所有 3 都统一变 7，而 1 骨架保留，因此 D6/3 的“颜色取反”也很直观。
#    - 规律完全建立在可见像素角色上，不依赖颜色编号大小、黑色排序或任何人类不可见因素。
# f) 可视化整洁度
#    - 没有额外标记层；每个对象仍然只是一个小型多色 sprite，且对象间留空明显。
#    - 输出只改色不挪动，输入输出对照很整洁，不会出现覆盖叠画造成的混乱。
# g) 颜色变化的可推断性
#    - 每题只涉及四种前景角色颜色：骨架 1、输入变量色 2/3、输出互补色 8/7。
#    - 两个输入变量色在所有实例中重复出现，并且总是成对地映到固定互补色；少量样例足以归纳映射。
# 8) 非退化量化指标
#    - `exact_mode_count`: 最小面积实例按完整像素数组应有 2 种模式。
#    - `color_set_mode_count`: 这些最小实例按“忽略排列只看颜色集合”应只有 1 种模式。
#      这两个量同时成立时，才能说明 D12/0 真的起作用而非退化回颜色集合分组。
#    - `legacy_diff_pixels`: 同一输入若按“原题式 exact-prototype 互换”产生的输出，与融合 solver 的输出必须
#      有正的像素差；否则就说明样本在行为上退化回原逻辑。
# 9) 新 solver 的隐式假设审查
#    - solver 假设所有需要改色的像素都处在非背景连通组件内；generator 不生成离散噪声像素。
#    - solver 假设颜色 1 是保留骨架色；generator 始终在两个 prototype 中放入 1，且 bg 永不取 1。
#    - solver 假设互补映射不会把前景映成背景；generator 的输入变量色固定为 2/3，而互补结果固定为 8/7。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve()
for _parent in ROOT.parents:
    solver_dir = _parent / "solvers"
    if (solver_dir / "arc_utils.py").exists():
        if str(solver_dir) not in sys.path:
            sys.path.append(str(solver_dir))
        break

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_COMPLEMENT = {
    0: 0,
    1: 1,
    2: 8,
    3: 7,
    4: 6,
    5: 9,
    6: 4,
    7: 3,
    8: 2,
    9: 5,
}


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of the color-1 scaffold inside a sprite array."""
    _ = bg
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "sprite array must contain color-1 anchor pixels"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _invert_color(color: int, bg: int) -> int:
    if color == bg:
        return bg
    return int(_COMPLEMENT.get(int(color), int(color)))


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
            }
        )
    return sprites


def _place_patch_aligned(
    out: np.ndarray,
    patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    inst_anchor: tuple[int, int, int, int],
    proto_anchor: tuple[int, int, int, int],
    scale: int,
    bg: int,
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = inst_anchor
    pr0, pc0, _pr1, _pc1 = proto_anchor
    top = (r0 + ar0) - pr0 * scale
    left = (c0 + ac0) - pc0 * scale

    ph, pw = patch.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    sub_p = patch[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    mask = sub_p != bg
    out_slice = out[rr0:rr1, cc0:cc1]
    out_slice[mask] = sub_p[mask]
    out[rr0:rr1, cc0:cc1] = out_slice


def _smallest_mode_stats(grid: np.ndarray, bg: int) -> tuple[int, int]:
    sprites = _extract_sprites(grid, bg)
    areas = [int(s["arr"].shape[0] * s["arr"].shape[1]) for s in sprites]
    min_area = min(areas)
    bases = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    exact_modes = {tuple(s["arr"].ravel().tolist()) for s in bases}
    color_set_modes = {
        tuple(sorted(int(v) for v in np.unique(s["arr"]) if int(v) not in (bg, 1)))
        for s in bases
    }
    return len(exact_modes), len(color_set_modes)


def _legacy_swap_output(grid: np.ndarray) -> np.ndarray:
    """Reference only for degeneracy checks: swap exact prototypes as in the original task."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprites = _extract_sprites(g, bg)

    areas = [int(s["arr"].shape[0] * s["arr"].shape[1]) for s in sprites]
    min_area = min(areas)
    bases = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in bases:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "legacy metric expects exactly two smallest exact prototypes"
    protos = list(uniq.values())

    base_anchor = _pick_anchor_bbox(protos[0], bg)
    base_anc_h = base_anchor[2] - base_anchor[0] + 1
    base_anc_w = base_anchor[3] - base_anchor[1] + 1

    out = g.copy()
    for s in sprites:
        arr = s["arr"]
        bbox = s["bbox"]
        inst_anchor = _pick_anchor_bbox(arr, bg)
        inst_anc_h = inst_anchor[2] - inst_anchor[0] + 1
        inst_anc_w = inst_anchor[3] - inst_anchor[1] + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, proto in enumerate(protos):
            scaled = _scale_nn(proto, scale)
            proto_anchor = _pick_anchor_bbox(proto, bg)
            pr0 = proto_anchor[0] * scale
            pc0 = proto_anchor[1] * scale
            top = (bbox[0] + inst_anchor[0]) - pr0
            left = (bbox[1] + inst_anchor[1]) - pc0
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(g.shape[0], top + scaled.shape[0])
            cc1 = min(g.shape[1], left + scaled.shape[1])
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            mask = sub_p != bg
            score = int(np.sum(sub_g[mask] == sub_p[mask])) if np.any(mask) else -1
            if score > best_score:
                best_score = score
                best_idx = idx

        repl = protos[1 - best_idx]
        for rr, cc in s["pts"]:
            out[rr, cc] = bg
        _place_patch_aligned(
            out,
            _scale_nn(repl, scale),
            bbox,
            inst_anchor,
            _pick_anchor_bbox(repl, bg),
            scale,
            bg,
        )
    return out


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    exact_mode_count, color_set_mode_count = _smallest_mode_stats(inp, bg)
    changed_pixels = int(np.sum((inp != out) & (inp != bg) & (inp != 1)))
    legacy_diff_pixels = int(np.sum(out != _legacy_swap_output(inp)))
    return {
        "exact_mode_count": int(exact_mode_count),
        "color_set_mode_count": int(color_set_mode_count),
        "changed_pixels": int(changed_pixels),
        "legacy_diff_pixels": int(legacy_diff_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every scaled sprite in place using a fixed complementary palette mapping.

    Input structure:
    - The background is the most common color. Foreground cells form 3-4 isolated sprites, each a nearest-neighbor scaled copy of one of two tiny prototypes.
    - The two prototype families deliberately share the same input color set and differ only by the exact pixel-level arrangement of those colors.
    - Every sprite contains color-1 scaffold pixels plus variable-color pixels drawn from a complementary palette pair.

    Rule:
    - Detect the sprite components so only real objects participate; the output keeps every sprite's footprint, scale, and position exactly unchanged.
    - Within each sprite, leave background and color 1 alone, and replace every other color by its fixed complement (in this dataset: 2->8 and 3->7).
    - Apply that recoloring cell-by-cell at the observed scale, so the precise spatial arrangement of the input colors is preserved while the palette flips.

    Key insight:
    - The old cross-family swap is gone: no sprite needs to borrow colors from another sprite or switch to the other prototype.
    - What matters is reading the visible pixel arrangement accurately, because the same 2/3 layout reappears as the exact same 8/7 layout after inversion.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sprites = _extract_sprites(g, bg)

    out = g.copy()
    for s in sprites:
        rr = s["pts"][:, 0]
        cc = s["pts"][:, 1]
        vals = out[rr, cc]
        out[rr, cc] = np.array([_invert_color(int(v), bg) for v in vals], dtype=int)
    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

PROTO_SHAPES = [(4, 3), (3, 4), (4, 4), (5, 4)]
BG_CHOICES = [0, 4, 5, 6, 9]
INPUT_COLORS = [2, 3]

ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, width: int) -> list[int]:
    return list(_ROW_MASKS[kind][width])


def _make_labels() -> np.ndarray:
    """Build a connected label grid using roles 1(anchor), 2, 3 and 0(bg holes)."""
    for _ in range(200):
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
            kinds, wts = zip(*weights_by_role[role])
            masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

        if not any(mask[0] or mask[-1] for mask in masks):
            idx = random.randrange(h)
            if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[idx] = _row_mask("outer", w)
            else:
                masks[idx] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            masks[random.randrange(h)] = _row_mask("center", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        if np.count_nonzero(labels == 1) == 0 or np.count_nonzero(labels == 2) == 0 or np.count_nonzero(labels == 3) == 0:
            continue
        if len(components_mask(labels != 0, connectivity=8)) != 1:
            continue
        if np.array_equal(labels, np.flipud(labels)):
            continue
        return labels
    raise RuntimeError("failed to build asymmetric labels")


def _colorize(labels: np.ndarray, bg: int) -> np.ndarray:
    proto = np.full(labels.shape, bg, dtype=int)
    proto[labels == 1] = 1
    proto[labels == 2] = INPUT_COLORS[0]
    proto[labels == 3] = INPUT_COLORS[1]
    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    proto_a = _colorize(labels, bg)
    proto_b = _colorize(np.flipud(labels), bg)
    assert tuple(sorted(int(v) for v in np.unique(proto_a) if int(v) not in (bg, 1))) == (2, 3)
    assert tuple(sorted(int(v) for v in np.unique(proto_b) if int(v) not in (bg, 1))) == (2, 3)
    assert not np.array_equal(proto_a, proto_b)
    return proto_a, proto_b


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(protos[idx].shape) * scale for idx, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None

    n = random.randint(min_n, GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for proto_idx, scale in instances:
        patch = _scale_nn(protos[proto_idx], scale)
        marker = np.full(patch.shape, 1, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(reserve, marker, r0, c0)
        paste(inp, patch, r0, c0, transparent=bg)

    try:
        out = main(inp)
    except Exception:
        return None

    metrics = _fusion_metrics(inp, out)
    if metrics["exact_mode_count"] != 2:
        return None
    if metrics["color_set_mode_count"] != 1:
        return None
    if metrics["changed_pixels"] <= 0:
        return None
    if metrics["legacy_diff_pixels"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
