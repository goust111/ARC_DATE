# Variant of 247ef758
# Dimension: D7 — 多步序列：覆盖顺序 (multi_step)
# Change: 按精灵在左侧条带中的位置从下到上排序（底部先印刻、顶部后印刻覆盖之前的）... → 从上到下排序（取反当前顺序）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D7 variant
# ----------------------------------
# variant_analysis:
#   concretization:
#     - 输入仍然是 "左侧精灵条带 | 单列分隔条 | 右侧矩形框"。
#     - 每个 mover 精灵仍按其颜色匹配框的左右行标记和上下列标记，并在行×列交点处居中复制。
#     - D7 唯一规则改动发生在 "多精灵复制到框内后如果互相覆盖，谁压谁"：
#       原题是下方精灵后印刻，因此它会盖住上方精灵；本变体改成上方精灵先印刻、下方精灵后印刻，
#       因而下方精灵成为最终盖住别人的对象。
#   behavior_trace:
#     - 小例子：左条带里有上方红精灵、下方蓝精灵；两者目标中心在框内相距 (1,1)，复制后有 3 个像素重叠。
#     - 原题顺序：蓝先印刻，红后印刻，重叠像素最终为红。
#     - 本变体顺序：红先印刻，蓝后印刻，重叠像素最终为蓝。
#     - 因此输入结构不变，差异只体现在输出中重叠区域的最终颜色。
#   complexity_calibration:
#     - 只反转排序会有退化风险：若目标彼此不重叠，则新旧输出完全相同，人类看不出 D7。
#     - 生成器因此强制至少一对 mover 在框内发生真实像素重叠，并要求新旧顺序得到不同输出。
#     - 其他 mover 允许保留原任务的 1-2 个行标记 / 1-2 个列标记，以保留笛卡尔积复制的支撑。
#   implicit_assumption_audit:
#     - `largest_enclosed_component_bbox(...)` 假设框内是最大的、且不接触边界的背景区域；生成器保留原框结构。
#     - divider 检测假设左侧存在几乎整列、单一颜色的非背景列；生成器继续放置整列分隔条。
#     - sprite 提取假设 mover 是左区里的单色 8 连通块；生成器继续使用单色 8 连通 blob。
#     - target 推断假设只有真正的 mover 颜色同时出现在成对边缘标记上；distractor 仍不出现在标记中。
#     - 排序反转后，`main()` 新增的关键前提是 mover 的上下顺序必须可见且不打平；生成器通过顺序堆叠保证中心行严格不同。
#     - 另一个新增前提是必须存在会受排序影响的覆盖；生成器用 "新旧输出必须不同" 的验证来保证。
#   placement_legality:
#     - 每个 target 中心都先按 sprite bbox 中心可落在 interior 内的范围采样，因此复制后不会越界。
#     - overlap pair 允许彼此重叠，但其余 mover 会避开已占用像素集合，避免视觉过乱并保护 D7 信号。
#     - 行标记和列标记在不同 mover 间仍不复用，避免匹配歧义。
#   observability:
#     - 变化只依赖可见的垂直位置顺序，绝不依赖颜色数值大小等隐含信息。
#     - 每个样本都要求 `top_to_bottom_output != bottom_to_top_output`，确保从样例中确实能观察到覆盖顺序反转。
#   visual_cleanliness:
#     - 只强制一对 mover 发生重叠，其余 mover 尽量不与已放置结果相交，避免整个框内大片混战。
#     - frame、divider、distractor 的视觉角色保持原样，因此人类仍能清楚分离 "控制标记" 与 "被搬运的精灵"。
#   color_reasoning:
#     - 颜色仍然只承担配对作用：左侧某颜色精灵对应框边同色标记。
#     - D7 不引入新的颜色映射，所以不会制造额外的颜色推断负担。
#   new_main_assumptions_checked_against_generator:
#     - `main()` 按 mover bbox 中心行排序；生成器的左侧堆叠不会产生同中心行并列。
#     - `main()` 仍默认所有 mover 都可在框内安全盖印；生成器只选取合法中心。
#     - `main()` 不显式检测 "顺序是否重要"；生成器用硬约束过滤掉不体现 D7 的样本。


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the grid border."""
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


def _solve(grid: np.ndarray, *, top_to_bottom: bool) -> np.ndarray:
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
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if top_to_bottom:
        # Variant D7: upper sprites stamp first; lower sprites stamp later and therefore cover.
        movers.sort(key=lambda s: (((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))
    else:
        # Original task order retained here only for generator-side visibility checks.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same structure as the original task, but the stamping order is reversed:
    movers are processed from top to bottom, so lower movers end up covering upper movers.
    """
    return _solve(grid, top_to_bottom=True)


def _d7_visibility_score(grid: np.ndarray) -> int:
    """How many output pixels change if we toggle the old vs new stamping order on the same input."""
    new_out = _solve(grid, top_to_bottom=True)
    old_out = _solve(grid, top_to_bottom=False)
    return int(np.count_nonzero(new_out != old_out))


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick `n` values from candidates with at least min_spacing between any two."""
    pool = list(candidates)
    if not pool or n < 1:
        return None
    if n == 1:
        return [random.choice(pool)]
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
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
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _sprite_rel_pixels(sprite: np.ndarray) -> set[tuple[int, int]]:
    center_r = (int(sprite.shape[0]) - 1) // 2
    center_c = (int(sprite.shape[1]) - 1) // 2
    return {(int(r) - center_r, int(c) - center_c) for r, c in np.argwhere(sprite != 0)}


def _translated_pixels(rel_pixels: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = map(int, center)
    return {(int(cr + dr), int(cc + dc)) for dr, dc in rel_pixels}


def _center_ranges(sprite: np.ndarray, ir0: int, ir1: int, ic0: int, ic1: int) -> tuple[list[int], list[int]] | None:
    sh, sw = map(int, sprite.shape)
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None
    return list(range(min_row, max_row + 1)), list(range(min_col, max_col + 1))


def _find_overlap_pair_assignment(
    sprite_a: np.ndarray,
    sprite_b: np.ndarray,
    rows_a: list[int],
    cols_a: list[int],
    rows_b: list[int],
    cols_b: list[int],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], int] | None:
    rel_a = _sprite_rel_pixels(sprite_a)
    rel_b = _sprite_rel_pixels(sprite_b)
    row_gap_limit = max(int(sprite_a.shape[0]), int(sprite_b.shape[0])) - 1
    col_gap_limit = max(int(sprite_a.shape[1]), int(sprite_b.shape[1])) - 1
    candidates = []

    for ra in rows_a:
        for rb in rows_b:
            if int(ra) == int(rb) or abs(int(ra) - int(rb)) > int(row_gap_limit):
                continue
            for ca in cols_a:
                pix_a = _translated_pixels(rel_a, (int(ra), int(ca)))
                for cb in cols_b:
                    if int(ca) == int(cb) or abs(int(ca) - int(cb)) > int(col_gap_limit):
                        continue
                    pix_b = _translated_pixels(rel_b, (int(rb), int(cb)))
                    overlap = int(len(pix_a & pix_b))
                    if overlap >= 2:
                        candidates.append((overlap, abs(int(ra) - int(rb)) + abs(int(ca) - int(cb)), (int(ra), int(ca)), (int(rb), int(cb)), pix_a, pix_b))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_overlap = int(candidates[0][0])
    shortlist = [item for item in candidates if int(item[0]) == best_overlap][:12]
    overlap, _gap, center_a, center_b, pix_a, pix_b = random.choice(shortlist)
    return center_a, center_b, pix_a, pix_b, int(overlap)


def _pick_nonoverlapping_marks(
    sprite: np.ndarray,
    row_candidates: list[int],
    col_candidates: list[int],
    occupied_rows: set[int],
    occupied_cols: set[int],
    blocked_pixel_sets: list[set[tuple[int, int]]],
) -> tuple[list[int], list[int], list[set[tuple[int, int]]]] | None:
    rel_pixels = _sprite_rel_pixels(sprite)
    sh, sw = map(int, sprite.shape)

    avail_rows = [int(r) for r in row_candidates if int(r) not in occupied_rows]
    avail_cols = [int(c) for c in col_candidates if int(c) not in occupied_cols]
    if not avail_rows or not avail_cols:
        return None

    for _ in range(80):
        want_rows = 2 if len(avail_rows) >= 2 and random.choice([False, True]) else 1
        want_cols = 2 if len(avail_cols) >= 2 and random.choice([False, True]) else 1
        row_marks = _pick_spaced(avail_rows, want_rows, sh)
        col_marks = _pick_spaced(avail_cols, want_cols, sw)
        if row_marks is None or col_marks is None:
            continue

        target_pixel_sets: list[set[tuple[int, int]]] = []
        legal = True
        for r in row_marks:
            for c in col_marks:
                pix = _translated_pixels(rel_pixels, (int(r), int(c)))
                if any(bool(pix & blocked) for blocked in blocked_pixel_sets):
                    legal = False
                    break
                target_pixel_sets.append(pix)
            if not legal:
                break
        if legal:
            return sorted(row_marks), sorted(col_marks), target_pixel_sets

    return None


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
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(c) for c in all_colors[idx: idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(c) for c in all_colors[idx: idx + n_distractors]]

    sprites: list[np.ndarray] = []
    ranges: list[tuple[list[int], list[int]]] = []
    for mc in mover_colors:
        sprite = _make_sprite(mc, 5, sprite_area_w)
        if sprite is None:
            return None
        center_ranges = _center_ranges(sprite, ir0, ir1, ic0, ic1)
        if center_ranges is None:
            return None
        sprites.append(sprite)
        ranges.append(center_ranges)

    pair_assignment = _find_overlap_pair_assignment(
        sprites[0],
        sprites[1],
        ranges[0][0],
        ranges[0][1],
        ranges[1][0],
        ranges[1][1],
    )
    if pair_assignment is None:
        return None
    center_a, center_b, pix_a, pix_b, overlap_pixels = pair_assignment
    if overlap_pixels < 2:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {
        int(mover_colors[0]): ([int(center_a[0])], [int(center_a[1])]),
        int(mover_colors[1]): ([int(center_b[0])], [int(center_b[1])]),
    }
    occupied_rows = {int(center_a[0]), int(center_b[0])}
    occupied_cols = {int(center_a[1]), int(center_b[1])}
    blocked_pixel_sets: list[set[tuple[int, int]]] = [pix_a, pix_b]

    for mc, sprite, center_ranges in zip(mover_colors[2:], sprites[2:], ranges[2:]):
        picked = _pick_nonoverlapping_marks(
            sprite,
            center_ranges[0],
            center_ranges[1],
            occupied_rows,
            occupied_cols,
            blocked_pixel_sets,
        )
        if picked is None:
            return None
        row_marks, col_marks, target_pixel_sets = picked
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))
        blocked_pixel_sets.extend(target_pixel_sets)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0): int(fc0) + frame_w] = top_edge
    inp[int(h - 1), int(fc0): int(fc0) + frame_w] = top_edge

    for r in range(1, h - 1):
        val = int(frame_base)
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if int(r) in set(map(int, rmarks)):
                val = int(mc)
                break
        inp[int(r), int(fc0)] = val
        inp[int(r), int(fc1)] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(int(mc), sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc_color), ds, True))

    for _color, sprite, _is_distractor in all_items:
        sh, sw = map(int, sprite.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sprite[int(dr), int(dc)]) != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sprite[int(dr), int(dc)])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        new_output = main(inp)
        old_output = _solve(inp, top_to_bottom=False)
    except Exception:
        return None

    if np.array_equal(inp, new_output):
        return None
    if int(np.count_nonzero(new_output != old_output)) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
