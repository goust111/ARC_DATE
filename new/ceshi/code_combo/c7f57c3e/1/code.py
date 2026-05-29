# Combo variant of c7f57c3e (index 1)
# combo_spec: D1/0+D2/2
# Name: 按形状分组 + 按洞数配对
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题把每个精灵直接识别成“两种彩色原型之一”，随后整块切换到另一原型。
#    - 这个融合版改成 4 个基础原型：2 个外轮廓家族 A/B，每个家族各有 1 孔、2 孔两个成员。
#    - 输入里同一家族共享同一组变量色，因此颜色仍有组信息；但真正的组键不再是颜色本身，而是
#      “把内部洞填上以后得到的外轮廓”。输出时每个精灵跳到另一形状家族里、洞数相同的那个成员。
#    - 例子：一个放大 3 倍的 A-2孔 精灵，旧逻辑会尝试按整体彩色原型去认；现在必须先把洞填上识别出
#      它属于 A 轮廓，再数出它有 2 个洞，最后只允许映射到 B 家族里的 2 孔成员，而不是 B-1孔。
# b) 复杂度校准
#    - 只保留 2 个形状家族和 2 档洞数 {1, 2}，总共 4 个基础原型，避免自由度过多。
#    - 每个样例固定出现 4 个 scale=1 字典精灵，各自把 4 个原型完整展示出来，再加 1 个放大精灵，
#      让“形状分组”和“洞数配对”都能在同一题里重复出现。
#    - 两个家族使用固定的两色调色板，但允许共享 1 个颜色，防止规则退化成“看见某色就换成另一色”。
# c) 隐式假设审查
#    - 原 main() 假设最小面积精灵只有 2 个原型；这里必须扩成恰好 4 个。
#    - 原 main() 直接按完整彩色阵列做匹配；这里若这么做，就无法体现 D1 的“按外轮廓分组”，所以改成
#      先 canonicalize，再把封闭洞填平后比较 shape signature。
#    - 原 main() 的替换目标是固定的“另一个原型”；这里不成立，必须先确定当前属于哪个形状家族，
#      再按洞数在对侧家族中查找对应成员。
#    - 新 solver 还假设所有基础原型共享同一 bbox 尺寸、同一 anchor 列和两个变量色槽位；
#      generator 用手工模板强制满足这些条件。
# d) 放置合法性
#    - 4 个基础原型和替换后的原型都使用同一个 5x5 bbox，且 anchor 位置固定，因此输入输出可用同一
#      左上角放置，不需要额外做并集 bbox 对齐。
#    - 生成器仍按大精灵优先放置，并用整块 bbox 做 reserve，保证不重叠、不越界、间距 >= 2。
# e) 视觉可观察性
#    - 人类先能看见有两类不同外轮廓；再能看见每类各有 1 孔与 2 孔两个成员；输出里每个对象都会变成
#      另一类外轮廓，但洞数保持不变。这个规则在 4 个基础小精灵上已经完整暴露，大精灵只是在更大尺度
#      上重复同一逻辑。
# f) 可视化整洁度
#    - 所有对象都只由一个连通精灵组成，没有覆盖叠加；内部洞只占少量位置，外轮廓清晰。
#    - 家族 A 的缺口在底边，家族 B 的缺口在右边，洞数变化只发生在内部，视觉层次稳定。
# g) 变化的可推断性
#    - 颜色变化总是伴随“形状家族切换”，不会无提示地乱改；洞数不决定颜色，只决定跨组后落到哪个成员。
#      因此解题者可以从结构线索而不是颜色编号大小去归纳映射。
# 9) 新 main() 的额外假设与 generator 保证
#    - 假设最小面积精灵完整覆盖 4 个基础原型：generator 固定放入 A1/A2/B1/B2 四个 scale=1 精灵。
#    - 假设每个形状家族恰好有两个洞数成员且洞数分别为 1 和 2：generator 使用固定模板保证。
#    - 假设最近邻缩放不改变封闭洞个数与填洞后的外轮廓：这里只使用整数 NN 放大，4 连通洞计数稳定。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, paste, random_free_location


HOLE_CONNECTIVITY = 4

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 32
GRID_MARGIN_FOR_SPRITES = 8

# --- CFG: palettes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.55

# --- CFG: sprite instances ---
BIG_SCALES = [2, 3]
MAX_RETRIES = 400

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _ordered_variable_colors(arr: np.ndarray, bg: int) -> list[int]:
    seen: list[int] = []
    for value in arr.ravel().tolist():
        v = int(value)
        if v in (bg, 1):
            continue
        if v not in seen:
            seen.append(v)
    assert len(seen) == 2, "each sprite must expose exactly two variable colors"
    return seen


def _canonicalize(arr: np.ndarray, bg: int) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=int)
    out[arr == 1] = 1
    ordered = _ordered_variable_colors(arr, bg)
    out[arr == ordered[0]] = 2
    out[arr == ordered[1]] = 3
    return out


def _count_enclosed_holes(arr: np.ndarray, bg: int) -> int:
    hole_mask = arr == bg
    lab, n = label_mask(hole_mask, connectivity=HOLE_CONNECTIVITY)
    h, w = arr.shape
    count = 0
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        count += 1
    return int(count)


def _filled_shape_mask(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != bg
    lab, n = label_mask(arr == bg, connectivity=HOLE_CONNECTIVITY)
    h, w = arr.shape
    filled = mask.copy()
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        filled[lab == k] = True
    return filled.astype(int)


def _paint(labels: np.ndarray, bg: int, palette: tuple[int, int]) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(palette[0])
    out[labels == 3] = int(palette[1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite shape family while preserving its hole count and scale.

    Input structure:
    - bg is the most common color. Four smallest sprites form a dictionary: two outer-contour families, and each family appears once with 1 enclosed hole and once with 2 enclosed holes.
    - Larger sprites are nearest-neighbor scaled copies of those dictionary members. Every sprite contains a fixed color-1 anchor spine plus exactly two variable-color regions.

    Rule:
    - Recover the four base prototypes from the smallest sprites, then group them by filled outer contour rather than by literal palette.
    - For each sprite, infer its scale, identify which contour family it belongs to after filling its enclosed holes, and count how many enclosed holes it has.
    - Replace that sprite with the prototype from the other contour family that has the same hole count, keeping the same scale and placement.

    Key insight:
    - The transformation uses two structural levels at once: outer contour decides the group, while enclosed-hole count decides the partner inside the opposite group. Colors alone are insufficient because the same palette is reused across the two hole-count members of each family.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 5, "grid must contain the 4 base prototypes plus one enlarged sprite"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 4, "smallest-area sprites must yield exactly 4 colored base prototypes"

    protos = list(uniq.values())
    proto_infos = []
    for proto in protos:
        canon = _canonicalize(proto, bg)
        proto_infos.append(
            {
                "arr": proto,
                "canon": canon,
                "shape_mask": _filled_shape_mask(canon, 0),
                "holes": _count_enclosed_holes(canon, 0),
            }
        )

    base_h, base_w = proto_infos[0]["canon"].shape
    for info in proto_infos[1:]:
        assert info["canon"].shape == (base_h, base_w), "all prototypes must share the same base bbox size"

    shape_groups: dict[tuple[int, ...], list[int]] = {}
    for idx, info in enumerate(proto_infos):
        sig = tuple(np.asarray(info["shape_mask"], dtype=int).ravel().tolist())
        shape_groups.setdefault(sig, []).append(idx)
    assert len(shape_groups) == 2 and all(len(v) == 2 for v in shape_groups.values()), "prototypes must form two shape families"

    families = []
    for sig, idxs in shape_groups.items():
        hole_to_idx = {int(proto_infos[idx]["holes"]): int(idx) for idx in idxs}
        assert sorted(hole_to_idx.keys()) == [1, 2], "each shape family must contain the 1-hole and 2-hole members"
        families.append(
            {
                "shape_mask": proto_infos[idxs[0]]["shape_mask"],
                "hole_to_idx": hole_to_idx,
            }
        )

    out = np.full_like(g, bg)

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = s["arr"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % base_h == 0 and w % base_w == 0, "sprite bbox must be an integer scale of the base prototypes"
        scale_h = h // base_h
        scale_w = w // base_w
        assert scale_h == scale_w, "scaling must be isotropic"
        scale = int(scale_h)

        canon = _canonicalize(arr, bg)
        shape_mask = _filled_shape_mask(canon, 0)
        holes = _count_enclosed_holes(canon, 0)
        assert holes in (1, 2), "every sprite must have 1 or 2 enclosed holes"

        current_family = None
        best_score = -1
        for fam_idx, fam in enumerate(families):
            scaled_mask = _scale_nn(np.asarray(fam["shape_mask"], dtype=int), scale)
            if scaled_mask.shape != shape_mask.shape:
                continue
            score = int(np.sum(scaled_mask == shape_mask))
            if score > best_score:
                best_score = score
                current_family = fam_idx
        assert current_family is not None, "sprite must match one shape family"

        other_family = 1 - current_family
        repl_idx = int(families[other_family]["hole_to_idx"][holes])
        replacement = _scale_nn(np.asarray(proto_infos[repl_idx]["arr"], dtype=int), scale)
        assert replacement.shape == arr.shape, "replacement should preserve bbox size"
        paste(out, replacement, r0, c0, transparent=bg)

    return out


# Shape family A: missing bottom corners. Shape family B: missing top-right and bottom-right corners.
# In both families, the 2-hole member differs only by adding one extra enclosed bg cell.
SHAPE_FAMILY_LABELS = {
    "A": {
        1: np.array(
            [
                [2, 2, 1, 3, 3],
                [2, 0, 1, 3, 3],
                [2, 2, 1, 3, 3],
                [2, 2, 1, 3, 3],
                [0, 2, 1, 3, 0],
            ],
            dtype=int,
        ),
        2: np.array(
            [
                [2, 2, 1, 3, 3],
                [2, 0, 1, 0, 3],
                [2, 2, 1, 3, 3],
                [2, 2, 1, 3, 3],
                [0, 2, 1, 3, 0],
            ],
            dtype=int,
        ),
    },
    "B": {
        1: np.array(
            [
                [2, 2, 1, 3, 0],
                [2, 2, 1, 3, 3],
                [2, 2, 1, 0, 3],
                [2, 2, 1, 3, 3],
                [2, 2, 1, 3, 0],
            ],
            dtype=int,
        ),
        2: np.array(
            [
                [2, 2, 1, 3, 0],
                [2, 2, 1, 0, 3],
                [2, 2, 1, 3, 3],
                [2, 2, 1, 0, 3],
                [2, 2, 1, 3, 0],
            ],
            dtype=int,
        ),
    },
}


def _pick_palettes(bg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    pool = [c for c in range(10) if c not in (bg, 1)]
    pal_a = tuple(random.sample(pool, 2))
    remaining = [c for c in pool if c not in pal_a]
    if random.random() < COLOR_SHARE_PROB:
        shared = random.choice(list(pal_a))
        fresh = random.choice(remaining)
        pal_b = (shared, fresh) if random.random() < 0.5 else (fresh, shared)
    else:
        pal_b = tuple(random.sample(remaining, 2))
    assert pal_a != pal_b
    return pal_a, pal_b


def _make_prototypes(bg: int) -> tuple[list[np.ndarray], dict[int, int]]:
    pal_a, pal_b = _pick_palettes(bg)
    ordered_keys = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    protos = []
    for family, holes in ordered_keys:
        palette = pal_a if family == "A" else pal_b
        protos.append(_paint(SHAPE_FAMILY_LABELS[family][holes], bg, palette))

    pair_map = {0: 2, 2: 0, 1: 3, 3: 1}
    assert sorted(_count_enclosed_holes(_canonicalize(p, bg), 0) for p in protos) == [1, 1, 2, 2]
    assert len({tuple(_filled_shape_mask(_canonicalize(p, bg), 0).ravel().tolist()) for p in protos}) == 2
    return protos, pair_map


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos, pair_map = _make_prototypes(bg)

    instances = [(idx, 1) for idx in range(4)]
    instances.append((random.randrange(4), random.choice(BIG_SCALES)))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    base_dim = protos[0].shape[0]
    max_dim = max(base_dim * scale for _idx, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_idx, scale in instances:
        output_patch = _scale_nn(protos[out_idx], scale)
        input_patch = _scale_nn(protos[pair_map[out_idx]], scale)
        marker = np.full(output_patch.shape, 1, dtype=int)
        try:
            top, left = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, top, left, transparent=bg)
        paste(inp, input_patch, top, left, transparent=bg)
        paste(out, output_patch, top, left, transparent=bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
