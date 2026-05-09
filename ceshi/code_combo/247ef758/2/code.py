# Combo variant of 247ef758 (index 2)
# combo_spec: D1/2+D9/1
# Name: 单侧匹配+部分边缘干扰

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for this combo variant
# a) 具体化与行为追踪
#    - 原题: 某个精灵颜色 C 必须同时满足两条约束才会移动:
#      1) 框架左/右边缘在同一行都出现 C;
#      2) 框架上/下边缘在同一列都出现 C。
#      然后把该精灵居中盖印到 row_marks(C) x col_marks(C) 的笛卡尔积上。
#    - 本融合版: 行维度改成“只看左边缘”，列维度仍保持“上下双边确认”。
#      也就是说，某色只要出现在左边某行，就拿到候选行；但只有同时在上/下边缘同列成对出现，
#      才拿到有效列。真正移动仍要求两个维度都成立。
#    - 具体小例子:
#      假设颜色 4 的精灵在左侧条带中。框架边缘如下:
#        左边第 5 行 = 4，右边第 5 行 = frame_base，
#        上/下第 9 列都 = 4。
#      原题会拒绝第 5 行，因为左右不相等；本融合版会接受第 5 行，并与第 9 列组成目标 (5, 9)。
#      再假设颜色 7 是干扰精灵，左边第 8 行 = 7，但上/下边缘没有任何成对的 7。
#      原题和本融合版都不会移动它，但本融合版里它“部分看起来像真行标记”，必须继续检查列维度。
#    - 可见差异:
#      mover 颜色能在“左边独有行标记 + 上下成对列标记”的情况下移动；
#      distractor 颜色也可能拥有左边行标记，但因没有成对列标记而保持不动。
#
# b) 复杂度校准
#    - 保留原题的整体结构: 左侧 2-3 个 mover 精灵 + 1 个 distractor，右侧一整个矩形框。
#    - 每个 mover 仍有 1-2 个行标记和 1-2 个列标记，行/列位置在不同颜色间不共享，
#      这样笛卡尔积复制的规律仍明显，而且不会退化成重叠难分。
#    - 为了让 D9/1 真的可见，生成器强制至少 1 个 distractor，且它的颜色至少出现在左边缘 1 次，
#      但绝不在上/下边缘形成任意一个成对列标记。
#    - 右边缘统一保持 frame_base，不再承担行确认，视觉上把“行只看左边”明确暴露出来，
#      同时也让“左边缘中的 distractor 颜色”更显眼。
#
# c) 隐式假设审查
#    - `largest_enclosed_component_bbox()` 仍假设框架内部是最大且不触边的背景连通块。
#      生成器保留原题的完整框体，因此该假设仍成立。
#    - divider 检测仍假设框架左侧存在一列几乎满高、统一颜色的分隔列。生成器继续固定这样构造。
#    - 左条带中的对象仍按“单色 8 连通分量”提取。生成器保持每个精灵单色、互不接触。
#    - 原 solver 最关键的旧假设是“有效行标记必须左右同色”。本融合版故意打破它，
#      因此 `main()` 的行标记提取必须只看左边，而不能再读右边做 gate。
#    - 新引入的 D9/1 会让 distractor 颜色也出现在左边缘，所以 `main()` 不能把“出现在边缘”直接等同于 mover；
#      它必须继续按颜色做 row_marks 与 col_marks 的交集。
#    - 由于列维度仍需要上下双边确认，生成器必须保证每个真正 mover 至少有 1 个成对列标记，
#      每个 distractor 则严格没有成对列标记。
#
# d) 放置合法性
#    - mover 的行/列标记都从其 sprite bbox 推出合法中心范围，确保复制到每个 target 后仍在框内；
#      即使 stamping 代码本身带 clipping，生成器也尽量构造“天然合法”的目标。
#    - 不同 mover 的行标记、列标记互不复用，并按 sprite 高/宽做最小间距约束，避免不同颜色复制后过度堆叠。
#    - 左侧条带中的精灵按自上而下堆叠放置，不重叠、不越界；distractor 保持原位。
#
# e) 视觉可观察性
#    - 在 2-4 个样例里，人眼可以稳定看到:
#      1) 右边缘基本不提供任何颜色信息；
#      2) mover 颜色在左边缘出现后会移动，但前提是顶部/底部也有同色列标记；
#      3) distractor 颜色虽然也出现在左边缘，却不会移动，因为缺少上下成对列标记。
#    - 这三点共同构成“二维使用不同确认规则，并且部分匹配会误导”的融合规律。
#
# f) 可视化整洁度
#    - 框架边缘只承载两类信息: 左边行标记、上/下列标记；右边只保留底色，避免四边同时杂糅。
#    - distractor 只在左边缘制造干扰，不再额外加顶部单边噪声，降低视觉噪音。
#    - mover 颜色数控制在 2-3，distractor 只 1 个，整体外观仍接近原题、但差异清晰。
#
# g) 变化的可推断性
#    - 没有颜色映射或数值大小规则，所有判断都来自“某颜色在边缘的可见位置关系”。
#    - 颜色总数保持少量且角色稳定: bg、divider、frame_base、2-3 个 mover、1 个 distractor。
#      解题者不需要猜颜色含义，只需观察哪些边缘位置被重复使用。
#
# h) 融合行为的量化标准
#    - 生成器内部会拒绝两类退化样本:
#      1) 若 combo 输出与原始“双侧行确认”输出完全相同，说明 D1/2 没有生效；
#      2) 若没有任何 distractor 颜色出现在左边缘、或它意外拥有了成对列标记，说明 D9/1 没有生效。
#    - 因此保留下来的样本必须同时满足:
#      `left_only_mover_rows > 0` 且 `partial_edge_distractors > 0`。
#
# i) 新 solver 的额外假设审查
#    - 新 solver 假设“左边缘的非背景 sprite 颜色都可以先视为候选行色”，但真正移动必须再和
#      “上下成对列色”做交集。生成器通过给 distractor 注入 left-only 行标记来验证这一假设。
#    - 新 solver 仍假设每个参与移动的颜色在左条带里恰好有一个 8 连通 sprite 分量。
#      生成器保证所有颜色唯一、对象不接触，避免同色多分量歧义。
#    - 新 solver 默认右边缘对行维度无语义。生成器把右边缘固定为 frame_base，从而显式满足该假设。


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """Return the bbox of the largest connected component that does not touch the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(w)[None, :] < int(fc0)

    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            crop = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": crop,
                }
            )

    return {
        "grid": g,
        "bg": bg,
        "h": int(h),
        "w": int(w),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "fr0": int(fr0),
        "fc0": int(fc0),
        "fr1": int(fr1),
        "fc1": int(fc1),
        "sprites": sprites,
        "move_colors": {int(s["color"]) for s in sprites},
    }


def _collect_marks(scene: dict, *, row_mode: str) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[tuple[int, int]]]]:
    g = scene["grid"]
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, (scene["ir0"], scene["ic0"], scene["ir1"], scene["ic1"]))
    fr0, fc0, fr1, fc1 = map(int, (scene["fr0"], scene["fc0"], scene["fr1"], scene["fc1"]))
    move_colors = set(map(int, scene["move_colors"]))

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if row_mode == "left_only":
            if cl != int(bg) and cl in move_colors:
                row_marks[int(cl)].append(int(r))
        elif row_mode == "mirrored":
            if cl == cr and cl != int(bg) and cl in move_colors:
                row_marks[int(cl)].append(int(r))
        else:
            raise ValueError(f"unknown row_mode: {row_mode}")

    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    return row_marks, col_marks, targets


def _stamp(scene: dict, targets: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    out = scene["grid"].copy()
    ir0, ic0, ir1, ic1 = map(int, (scene["ir0"], scene["ic0"], scene["ir1"], scene["ic1"]))
    bg = int(scene["bg"])

    movers = [s for s in scene["sprites"] if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _solve_with_row_mode(grid: np.ndarray, *, row_mode: str) -> np.ndarray:
    scene = _extract_scene(grid)
    _row_marks, _col_marks, targets = _collect_marks(scene, row_mode=row_mode)
    return _stamp(scene, targets)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Row marks are confirmed by the LEFT frame edge only.
    - Column marks still require mirrored TOP/BOTTOM confirmation.
    - Distractor colors may appear on the left edge, but they still do not move unless
      they also have confirmed columns on both top and bottom edges.
    """
    return _solve_with_row_mode(grid, row_mode="left_only")


def _solve_original_semantics(grid: np.ndarray) -> np.ndarray:
    """Reference solver for generation-time anti-degeneracy checks."""
    return _solve_with_row_mode(grid, row_mode="mirrored")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    if max_h < 1 or max_w < 1:
        return None
    shape_choices = [
        (int(sh), int(sw))
        for sh in range(1, min(5, int(max_h)) + 1)
        for sw in range(1, min(5, int(max_w)) + 1)
        if int(sh * sw) >= 3
    ]
    if not shape_choices:
        return None
    sh, sw = random.choice(shape_choices)
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = int(color)
    return sprite


def _sample_sprite_set(colors: list[int], h: int, sprite_area_w: int) -> dict[int, np.ndarray] | None:
    sprites: dict[int, np.ndarray] = {}
    remaining_height = int(h)
    remaining_items = int(len(colors))
    for color in colors:
        max_h = min(5, remaining_height - (remaining_items - 1))
        sprite = _make_sprite(int(color), max_h, sprite_area_w)
        if sprite is None:
            return None
        sprites[int(color)] = sprite
        remaining_height -= int(sprite.shape[0])
        remaining_items -= 1
    return sprites


def _combo_metrics(grid: np.ndarray, distractor_colors: set[int]) -> dict[str, int | bool]:
    scene = _extract_scene(grid)
    row_left, col_both, targets_left = _collect_marks(scene, row_mode="left_only")
    row_both, _col_both_orig, targets_orig = _collect_marks(scene, row_mode="mirrored")

    g = scene["grid"]
    fc0 = int(scene["fc0"])
    fc1 = int(scene["fc1"])

    left_only_mover_rows = 0
    partial_edge_distractors = 0
    distractor_candidate_rows = 0

    for color in scene["move_colors"]:
        rows_left = row_left.get(int(color), [])
        rows_both = row_both.get(int(color), [])
        left_only_rows = [r for r in rows_left if r not in rows_both]
        if int(color) not in distractor_colors:
            left_only_mover_rows += int(len(left_only_rows))
        else:
            distractor_candidate_rows += int(len(rows_left))
            if rows_left and not col_both.get(int(color)):
                if all(int(g[int(r), int(fc1)]) != int(color) for r in rows_left):
                    if all(int(g[int(r), int(fc0)]) == int(color) for r in rows_left):
                        partial_edge_distractors += 1

    combo_out = _stamp(scene, targets_left)
    orig_out = _stamp(scene, targets_orig)
    return {
        "left_only_mover_rows": int(left_only_mover_rows),
        "partial_edge_distractors": int(partial_edge_distractors),
        "distractor_candidate_rows": int(distractor_candidate_rows),
        "combo_differs_from_original": bool(not np.array_equal(combo_out, orig_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = int(sprite_area_w)
    fc0, fc1 = int(sprite_area_w + 1), int(w - 1)
    ir0, ic0 = 1, int(fc0 + 1)
    ir1, ic1 = int(h - 2), int(fc1 - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = 1
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(c) for c in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(c) for c in all_colors[idx : idx + n_distractors]]

    all_sprite_colors = mover_colors + distractor_colors
    sprite_map = _sample_sprite_set(all_sprite_colors, h, sprite_area_w)
    if sprite_map is None:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc in mover_colors:
        sp = sprite_map[int(mc)]
        sh, sw = map(int, sp.shape)
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(1, sh))
        if not row_marks:
            return None
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), max(1, sw))
        if not col_marks:
            return None
        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))

    distractor_targets: dict[int, list[int]] = {}
    for dc in distractor_colors:
        sp = sprite_map[int(dc)]
        sh = int(sp.shape[0])
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        if min_row > max_row:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(1, sh))
        if not row_marks:
            return None
        distractor_targets[int(dc)] = sorted(row_marks)
        occupied_rows.update(map(int, row_marks))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = top_edge

    left_edge = np.full(h - 2, frame_base, dtype=int)
    for mc, (rmarks, _cmarks) in mover_targets.items():
        for rr in rmarks:
            left_edge[int(rr - 1)] = int(mc)
    for dc, rmarks in distractor_targets.items():
        for rr in rmarks:
            left_edge[int(rr - 1)] = int(dc)
    inp[1 : h - 1, fc0] = left_edge
    inp[1 : h - 1, fc1] = int(frame_base)

    cur_row = 0
    all_items = [(int(color), sprite_map[int(color)]) for color in all_sprite_colors]
    random.shuffle(all_items)
    for color, sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(color)
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        combo_out = main(inp)
        original_out = _solve_original_semantics(inp)
        metrics = _combo_metrics(inp, set(map(int, distractor_colors)))
    except Exception:
        return None

    if np.array_equal(inp, combo_out):
        return None
    if np.array_equal(combo_out, original_out):
        return None
    if int(metrics["left_only_mover_rows"]) <= 0:
        return None
    if int(metrics["partial_edge_distractors"]) <= 0:
        return None
    if int(metrics["distractor_candidate_rows"]) <= 0:
        return None
    if not bool(metrics["combo_differs_from_original"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
