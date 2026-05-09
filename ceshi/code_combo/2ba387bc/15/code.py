# Combo variant of 2ba387bc (index 15)
# combo_spec: D4/0+D9/0
# Name: 彩色内部框架+同色方块
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是若干散落在背景上的 4x4 对象，输出仍把 frame 堆到左列、solid 堆到右列。
#    - 融合后 frame 的像素结构是：
#        b b b b
#        b i i b
#        b i i b
#        b b b b
#      其中边框色 b 与中心色 i 都是非背景色，且 b != i。
#      solid 仍是 4x4 全同色：
#        s s s s
#        s s s s
#        s s s s
#        s s s s
#    - 小例子：
#      原题里，一个 4x4 对象若中心 2x2 是背景色，则被识别为 frame；否则被原 solver 归到 solid。
#      融合后，一个“红框蓝芯”的对象中心不是背景色，旧逻辑会误判为 solid；
#      新逻辑必须读出“边框单色 + 中心单色 + 二者不同”，才能把它判为 frame。
#      同时若再放一个纯红 solid，颜色不再能帮忙区分类别，因为红色既可能是 frame 的边框，
#      也可能是 solid 的主体。
# b) 复杂度校准
#    - 只改对象识别与颜色复用，不改尺寸、排序、输出布局，避免引入额外自由度。
#    - 生成器使用 2..4 色的小色盘，并强制至少一个 solid 的颜色出现在某个 frame 的边框或内芯中，
#      使“同色不同形”每个样本都可见，而不是偶然发生。
#    - 颜色复用不仅发生在跨类型之间，也允许多个 frame 共享边框色或内芯色，防止 solver 退化为
#      “每种颜色只对应一种结构”。
# c) 隐式假设审查
#    - `get_bg(g)` 仍成立：背景仍是网格中最多的颜色。
#    - `find_components(..., monochromatic=False)` 假设每个对象是独立的 4 邻接前景连通块。
#      融合后 frame 因为中心非背景，成为完整 16 格连通块；这个假设仍成立。
#    - 原代码的 `inner == bg -> frame` 前提被 D4 直接打破，必须换成边框色/中心色关系判断。
#    - 原代码把“不是 frame 的 4x4”一律当作 solid；融合后为了避免误收第三类模式，
#      这里改成显式检测 `16 格全同且非背景` 才算 solid。
#    - 排序仍依赖 `(row, col)`，与颜色值无关；生成器不允许任何需要按颜色排序的隐藏规则。
# d) 放置合法性
#    - 生成器只构造 4x4 patch，并用 `random_free_location(..., border=1, padding=1, connectivity=4)`
#      放置，保证对象不重叠、不越界、且 4 邻接下彼此分离。
#    - 因为 frame 与 solid 都没有背景洞，放置时用普通 `paste` 即可；分离性完全由 padding 保证。
# e) 视觉可观察性
#    - 人类可以直接看到：左列是“边框一种颜色、中心另一种颜色”的双色块，右列是纯色块。
#    - 同一个颜色会同时出现在 frame 和 solid 上，所以解题者会被迫看内部结构，而不能看颜色身份。
#    - 旧题的“中心是背景洞”在线索上被明确替换成“中心色与边框色不同”，从少量样例即可观察到。
# f) 可视化整洁度
#    - 所有对象都还是 4x4 规则方块，不叠加额外标记，也不引入旋转镜像。
#    - 对象之间至少隔 1 个背景单元，既能稳定分连通块，也能保持画面清楚。
# g) 变化的可推断性
#    - 没有颜色映射或数值运算；颜色只原样复制到输出。
#    - 可推断的核心是“关系”而不是“具体值”：frame 看边框色是否与中心色不同，solid 看是否全同。
#    - 为了让 D9 真正显性化，生成器强制 `frame_visible_colors ∩ solid_colors != ∅`。
# 8) 融合行为的量化指标
#    - `shared_cross_type_colors`: 同时出现在 frame 可见颜色集合（边框或中心）与 solid 颜色集合中的颜色数。
#      该值至少为 1，表示 D9/0 真的发生。
#    - `legacy_matches`: 旧 solver（把中心背景洞当 frame）在同一输入上的输出是否与新 solver 相同。
#      该值必须为 False，表示 D4/0 没有退化回原题。
# 9) main 的新隐式假设与 generator 保证
#    - 假设每个有效对象 bbox 都是恰好 4x4：generator 只生成 4x4 patch。
#    - 假设 frame 的 12 个边框格同色、中心 2x2 同色且二者不同：`_make_frame` 保证。
#    - 假设 solid 是全同色且非背景：`_make_solid` 保证。
#    - 假设不存在第三类 4x4 模式：generator 不生成其他图案，且自检会拒绝识别不一致的样本。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


def _frame_signature(sub: np.ndarray) -> tuple[int, int] | None:
    if sub.shape != (4, 4):
        return None
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    inner = sub[1:3, 1:3]
    border_colors = set(map(int, border.flatten()))
    inner_colors = set(map(int, inner.flatten()))
    if len(border_colors) != 1 or len(inner_colors) != 1:
        return None
    border_color = int(next(iter(border_colors)))
    inner_color = int(next(iter(inner_colors)))
    if border_color == inner_color:
        return None
    return border_color, inner_color


def _is_frame_patch(sub: np.ndarray, bg: int) -> bool:
    sig = _frame_signature(sub)
    if sig is None:
        return False
    border_color, inner_color = sig
    return border_color != int(bg) and inner_color != int(bg)


def _is_solid_patch(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    colors = set(map(int, sub.flatten()))
    return len(colors) == 1 and int(next(iter(colors))) != int(bg)


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_frame_patch(sub, bg):
            frames.append((int(r0), int(c0), sub))
        elif _is_solid_patch(sub, bg):
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _stack_objects(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by structure.

    Frames are 4x4 patches with a uniform 1px border and a uniform 2x2 center of a different
    non-background color. Solids are 4x4 single-color non-background squares. Colors may repeat
    across and within categories, so classification must come from structure rather than color IDs.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _stack_objects(frames, solids, bg)


def _legacy_is_frame_patch(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _legacy_is_frame_patch(sub, bg):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_objects(frames, solids, bg)


def _make_frame(border_col: int, inner_col: int) -> np.ndarray:
    patch = np.full((4, 4), int(border_col), dtype=int)
    patch[1:3, 1:3] = int(inner_col)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _rand_palette(total_objects: int) -> list[int]:
    max_size = min(4, max(2, total_objects - 1), 9)
    palette_size = int(np.random.randint(2, max_size + 1))
    return list(map(int, np.random.choice(np.arange(1, 10), size=palette_size, replace=False)))


def _sample_frame_specs(n_frames: int, palette: list[int], shared_color: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []

    shared_role = str(np.random.choice(["border", "inner"]))
    if shared_role == "border":
        border_col = int(shared_color)
        inner_choices = [c for c in palette if c != border_col]
        inner_col = int(np.random.choice(inner_choices))
    else:
        inner_col = int(shared_color)
        border_choices = [c for c in palette if c != inner_col]
        border_col = int(np.random.choice(border_choices))
    specs.append((border_col, inner_col))

    for i in range(1, n_frames):
        if i == 1 and n_frames >= 2 and np.random.rand() < 0.7:
            border_col = int(specs[0][0])
            inner_choices = [c for c in palette if c != border_col]
            inner_col = int(np.random.choice(inner_choices))
        else:
            border_col = int(np.random.choice(palette))
            inner_choices = [c for c in palette if c != border_col]
            inner_col = int(np.random.choice(inner_choices))
        specs.append((border_col, inner_col))

    return specs


def _sample_solid_colors(n_solids: int, palette: list[int], shared_color: int) -> list[int]:
    colors = [int(shared_color)]
    for _ in range(n_solids - 1):
        colors.append(int(np.random.choice(palette)))
    return colors


def _analyze_input(g: np.ndarray) -> dict[str, int | bool]:
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)

    frame_border_colors = [int(sub[0, 0]) for _r0, _c0, sub in frames]
    frame_inner_colors = [int(sub[1, 1]) for _r0, _c0, sub in frames]
    frame_visible_colors = set(frame_border_colors) | set(frame_inner_colors)
    solid_colors = [int(sub[0, 0]) for _r0, _c0, sub in solids]

    repeated_frame_border_colors = 0
    if frame_border_colors:
        repeated_frame_border_colors = int(len(frame_border_colors) - len(set(frame_border_colors)))

    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "frame_centers_non_bg": int(sum(int(sub[1, 1] != bg) for _r0, _c0, sub in frames)),
        "shared_cross_type_colors": int(len(frame_visible_colors & set(solid_colors))),
        "repeated_frame_border_colors": int(repeated_frame_border_colors),
        "legacy_matches": bool(np.array_equal(main(g), _legacy_main(g))),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    total_objects = int(n_frames + n_solids)
    palette = _rand_palette(total_objects)
    shared_color = int(np.random.choice(palette))
    frame_specs = _sample_frame_specs(n_frames, palette, shared_color)
    solid_colors = _sample_solid_colors(n_solids, palette, shared_color)

    try:
        frames: list[tuple[int, int, np.ndarray]] = []
        solids: list[tuple[int, int, np.ndarray]] = []

        for border_col, inner_col in frame_specs:
            patch = _make_frame(border_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, int(r0), int(c0))
            frames.append((int(r0), int(c0), patch))

        for col in solid_colors:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, int(r0), int(c0))
            solids.append((int(r0), int(c0), patch))
    except Exception:
        return None

    expected = _stack_objects(
        sorted(frames, key=lambda t: (int(t[0]), int(t[1]))),
        sorted(solids, key=lambda t: (int(t[0]), int(t[1]))),
        bg,
    )
    actual = main(g)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    info = _analyze_input(g)
    if info["n_frames"] != n_frames or info["n_solids"] != n_solids:
        return None
    if info["frame_centers_non_bg"] != n_frames:
        return None
    if int(info["shared_cross_type_colors"]) < 1:
        return None
    if bool(info["legacy_matches"]):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
