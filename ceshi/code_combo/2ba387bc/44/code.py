# Combo variant of 2ba387bc (index 44)
# combo_spec: D1/0+D2/2+D13/2
# Name: 空间分离陷阱+距离排序+边框分类
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题输入里是多个散落的 4x4 物体，通常可分成“空心框架”和“实心方块”，输出把两类
#      分别堆到左右两列，并按输入行序排序。
#    - 本融合版仍然只使用 4x4 物体，但输入被故意布置成：
#        1. 上半部分只放“框架形”物体：4x4 外环非背景、2x2 中心是背景。
#        2. 下半部分只放“实心形”物体：4x4 中心不是背景。
#      这样观察者很容易先误以为“上面一类、下面一类”就是分组依据。
#    - 真正分组却改成 D1/0：只看外边框 12 个像素是否同色。
#        * 单色边框 -> 左列 A 组。
#        * 多色边框 -> 右列 B 组。
#      于是上半部的两个框架形物体会分属左右列，下半部的两个实心形物体也会分属左右列，
#      位置与内部是否有洞都不再决定列归属。
#    - 组内顺序再叠加 D2/2：按物体中心到整张网格中心的欧氏距离由近到远排序，而不是按
#      `(row, col)`。例如：
#        * 左列若包含一个远离中心的上方框架和一个靠近中心的下方实心块，
#          原题行序会把上方框架排前；本题距离序会把靠近中心的下方实心块排前。
#    - 因而融合后的差异是双重的：
#        * 分类不再跟“上半/下半”或“有洞/无洞”一致。
#        * 排序也不再跟“从上到下”一致，而要比较到全局中心的距离。
# b) 复杂度校准
#    - 仍然只使用 4x4 孤立物体，避免对象类型膨胀。
#    - 边框只分两档：单色 / 多色；多色边框统一采用“顶边 + 右边换成强调色”的模式，
#      视觉上足够稳定，不会出现每个样例都像不同规则。
#    - 每个样例固定包含四个关键物体：
#        * 上半部：1 个单色边框框架 + 1 个多色边框框架
#        * 下半部：1 个单色边框实心块 + 1 个多色边框实心块
#      这能同时提供“位置误导”“内部形状误导”“边框分类真规则”的重复支撑。
#    - 距离排序至少在一组上强制打破原来的行序，避免样例退化成原题。
# c) 隐式假设审查（逐段对应 main）
#    - `find_components(... monochromatic=False)` 依赖：物体即使多色，也必须是 4 连通的单个组件。
#      生成器里的多色框架和多色实心块都保持整块连通，因此该前提成立。
#    - `bbox_h == bbox_w == 4` 依赖：每个目标物体仍是严格 4x4。生成器固定只放 4x4 patch。
#    - 原题里“看 2x2 内部是否为背景”是关键判定；本题中该前提失效，不能再用它决定列归属。
#      solver 改为只读取外环颜色集合。
#    - 新的 `_has_uniform_border(sub)` 依赖：外环 12 格完整可见、没有别的物体粘连污染边框。
#      生成器通过整块 4x4 + 四周 1 格空白保证这一点。
#    - 距离排序引入新的温和假设：同一组里最好不要恰好等距，否则主规则不明显。
#      solver 用 `(dist2, row, col)` 打破平局；generator 额外拒绝同组等距样例。
#    - 输出仍是假定“左列和右列各自按 4x4 堆叠、宽度固定 8”。这与原题兼容，未被改动破坏。
# d) 放置合法性
#    - 生成器只在合法行带内采样位置：框架形始终放在上半区，实心形始终放在下半区。
#    - 每个 4x4 物体都要求离边界至少 1 格，且其 4x4 bbox 外围 1 格的 6x6 区域全为空，
#      因而不会重叠、不会越界、不会粘连。
# e) 视觉可观察性
#    - 人类肉眼可见的线索有三个：上/下空间分离、是否有洞、边框是否单色、与中心的远近。
#    - 真规则只依赖后两者：边框一致性决定列，离中心远近决定组内顺序。
#    - 由于每一列都同时包含一个上方框架形物体和一个下方实心形物体，解题者能排除
#      “按上半/下半分组”的错误假设。
#    - 由于至少一列的顺序与原始行序不同，解题者也能排除“从上到下堆叠”的旧规则。
# f) 可视化整洁度
#    - 输入始终是 4 个孤立 4x4 物体，背景干净，没有额外标记或杂点。
#    - 多色边框只使用两种颜色，且模式固定，避免视觉噪声。
#    - 输出仍维持原题的两列堆叠格式，所以难点集中在“如何分类/排序”，而不是画面混乱。
# g) 变化的可推断性
#    - 没有任意颜色映射；所有颜色只是对象自身图案的一部分，输出完全保留对象原貌。
#    - 颜色种类控制在 ARC 基本调色板内，每个对象拥有清晰可辨的主色，强调色只用于制造
#      “多色边框”这一结构性特征。
# 9) main() 的新隐式假设与 generator 保证
#    - 假设：所有相关对象都是孤立 4x4 连通块。
#      保证：生成器只放置四个满足该条件的 patch，并验证 `main(g)` 可运行。
#    - 假设：单色边框组和多色边框组都至少非空，且最好各有两个对象以便观察排序。
#      保证：生成器固定每组各放两个对象。
#    - 假设：至少一组的距离排序和旧行序不同，否则 D2/2 不可见。
#      保证：生成器构造“上方远 / 下方近”与“上方近 / 下方远”的角色布局，并用可见性检查拒绝退化样例。
#    - 假设：边框分类真的跨越“框架形/实心形”两种外观。
#      保证：两组都同时包含上方框架形与下方实心形对象。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


def _border_pixels(sub: np.ndarray) -> np.ndarray:
    return np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])


def _has_uniform_border(sub: np.ndarray) -> bool:
    return len(set(map(int, _border_pixels(sub).flatten()))) == 1


def _is_frame_shaped(sub: np.ndarray, bg: int) -> bool:
    return bool((sub[1:3, 1:3] == int(bg)).all())


def _distance2_to_center(r0: int, c0: int, H: int, W: int) -> int:
    """
    Squared Euclidean distance on doubled coordinates.

    Object center: (r0 + 1.5, c0 + 1.5)
    Grid center:   ((H - 1) / 2, (W - 1) / 2)
    """
    dr2 = int(2 * int(r0) + 3 - (int(H) - 1))
    dc2 = int(2 * int(c0) + 3 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _collect_objects(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    uniforms: list[dict] = []
    mixeds: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        obj = {
            "r0": int(r0),
            "c0": int(c0),
            "sub": sub.copy(),
            "dist2": _distance2_to_center(int(r0), int(c0), int(H), int(W)),
            "uniform_border": _has_uniform_border(sub),
            "frame_shaped": _is_frame_shaped(sub, bg),
        }
        if bool(obj["uniform_border"]):
            uniforms.append(obj)
        else:
            mixeds.append(obj)

    return int(bg), uniforms, mixeds


def _distance_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"])))


def _row_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))


def _render(bg: int, left_group: list[dict], right_group: list[dict]) -> np.ndarray:
    outH = 4 * max(int(len(left_group)), int(len(right_group)))
    out = np.full((outH, 8), int(bg), dtype=int)
    for i, obj in enumerate(left_group):
        paste(out, obj["sub"], int(i * 4), 0)
    for i, obj in enumerate(right_group):
        paste(out, obj["sub"], int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects into two columns by border consistency.

    Variant rule:
    - Left column:  objects whose outer 12-pixel border is monochrome.
    - Right column: objects whose outer border contains multiple colors.
    - Inside each group, sort by center-distance to the whole-grid center.
    - Ties are broken by (row, col) only for determinism.
    """
    g = np.array(grid, dtype=int)
    bg, uniforms, mixeds = _collect_objects(g)
    assert uniforms or mixeds
    return _render(bg, _distance_sorted(uniforms), _distance_sorted(mixeds))


def _make_uniform_frame(border_col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(border_col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_mixed_frame(primary: int, accent: int, bg: int = 0) -> np.ndarray:
    a = _make_uniform_frame(primary, bg)
    a[0, :] = int(accent)
    a[:, 3] = int(accent)
    a[1:3, 1:3] = int(bg)
    return a


def _make_uniform_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_mixed_solid(primary: int, accent: int) -> np.ndarray:
    a = np.full((4, 4), int(primary), dtype=int)
    a[0, :] = int(accent)
    a[:, 3] = int(accent)
    return a


def _sample_colors() -> tuple[list[int], list[int]]:
    palette = list(map(int, np.random.permutation(np.arange(1, 10))))
    primaries = palette[:4]
    accents = palette[4:8]
    if len(accents) < 4:
        accents = (accents + primaries)[:4]
    return primaries, accents


def _all_locations(g: np.ndarray, row_lo: int, row_hi: int, bg: int = 0) -> list[tuple[int, int]]:
    H, W = map(int, g.shape)
    h = 4
    w = 4
    r_min = max(1, int(row_lo))
    r_max = min(int(row_hi), int(H - h - 1))
    c_min = 1
    c_max = int(W - w - 1)
    if r_max < r_min or c_max < c_min:
        return []

    candidates: list[tuple[int, int]] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            if bool((g[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] != int(bg)).any()):
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _pick_location(
    g: np.ndarray,
    row_lo: int,
    row_hi: int,
    *,
    col_bias: str,
    bg: int = 0,
) -> tuple[int, int]:
    H, W = map(int, g.shape)
    candidates = _all_locations(g, int(row_lo), int(row_hi), int(bg))
    if not candidates:
        raise ValueError("no valid location in requested band")

    center_c2 = int(W - 1)

    def horiz_key(rc: tuple[int, int]) -> int:
        _r0, c0 = rc
        return abs(int(2 * int(c0) + 3 - center_c2))

    candidates = sorted(candidates, key=horiz_key, reverse=(col_bias == "far"))
    keep = max(1, len(candidates) // 2)
    chosen = candidates[:keep]
    idx = int(np.random.randint(0, len(chosen)))
    return tuple(map(int, chosen[idx]))


def _place(g: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int = 0) -> None:
    paste(g, patch, int(r0), int(c0), transparent=int(bg))


def _legacy_output(g: np.ndarray) -> np.ndarray:
    bg, uniforms, mixeds = _collect_objects(np.array(g, dtype=int))
    legacy_frames = [obj for obj in uniforms if bool(obj["frame_shaped"])]
    legacy_solids = [obj for obj in uniforms if not bool(obj["frame_shaped"])] + list(mixeds)
    return _render(bg, _row_sorted(legacy_frames), _row_sorted(legacy_solids))


def _border_row_output(g: np.ndarray) -> np.ndarray:
    bg, uniforms, mixeds = _collect_objects(np.array(g, dtype=int))
    return _render(bg, _row_sorted(uniforms), _row_sorted(mixeds))


def _position_signature(group: list[dict], *, distance: bool) -> tuple[tuple[int, int], ...]:
    ordered = _distance_sorted(group) if distance else _row_sorted(group)
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _variant_visible(g: np.ndarray) -> bool:
    bg, uniforms, mixeds = _collect_objects(np.array(g, dtype=int))
    H, _W = map(int, g.shape)
    groups = [uniforms, mixeds]

    if any(len(group) != 2 for group in groups):
        return False
    if any(len({int(obj["dist2"]) for obj in group}) != len(group) for group in groups):
        return False

    center_r2 = int(H - 1)
    for group in groups:
        if not any(bool(obj["frame_shaped"]) for obj in group):
            return False
        if not any(not bool(obj["frame_shaped"]) for obj in group):
            return False

    for obj in uniforms + mixeds:
        obj_center_r2 = int(2 * int(obj["r0"]) + 3)
        if bool(obj["frame_shaped"]) and obj_center_r2 >= center_r2:
            return False
        if (not bool(obj["frame_shaped"])) and obj_center_r2 <= center_r2:
            return False

    if _position_signature(uniforms, distance=False) == _position_signature(uniforms, distance=True):
        return False

    combo_out = main(g)
    if np.array_equal(combo_out, _legacy_output(g)):
        return False
    if np.array_equal(combo_out, _border_row_output(g)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 31))
    W = int(np.random.randint(24, 31))
    g = np.full((H, W), int(bg), dtype=int)

    top_max = int(H // 2 - 4)
    bottom_min = int(H // 2 + 1)
    if top_max < 4 or bottom_min > H - 6:
        return None

    top_far_lo = 1
    top_far_hi = int(top_max - 2)
    top_near_lo = int(top_max - 1)
    top_near_hi = int(top_max)
    bottom_near_lo = int(bottom_min)
    bottom_near_hi = int(bottom_min + 1)
    bottom_far_lo = int(bottom_min + 2)
    bottom_far_hi = int(H - 5)
    if min(top_far_hi - top_far_lo, top_near_hi - top_near_lo, bottom_near_hi - bottom_near_lo, bottom_far_hi - bottom_far_lo) < 0:
        return None

    primaries, accents = _sample_colors()
    patches = [
        (_make_uniform_frame(primaries[0], bg), (top_far_lo, top_far_hi), "far"),
        (_make_mixed_frame(primaries[1], accents[1], bg), (top_near_lo, top_near_hi), "near"),
        (_make_uniform_solid(primaries[2]), (bottom_near_lo, bottom_near_hi), "near"),
        (_make_mixed_solid(primaries[3], accents[3]), (bottom_far_lo, bottom_far_hi), "far"),
    ]

    try:
        for patch, (row_lo, row_hi), col_bias in patches:
            r0, c0 = _pick_location(g, int(row_lo), int(row_hi), col_bias=col_bias, bg=bg)
            _place(g, patch, int(r0), int(c0), bg)

        if not _variant_visible(g):
            return None

        _ = main(g)
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
