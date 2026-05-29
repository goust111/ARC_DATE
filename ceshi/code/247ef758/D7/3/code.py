# Variant of 247ef758
# Dimension: D7 — 多步序列：覆盖顺序 (multi_step)
# Change: 按精灵在左侧条带中的位置从下到上排序（底部先印刻、顶部后印刻覆盖之前的）... → 按标记数量排序

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis for the D7 variant
==================================

variant_spec:
  unchanged_layout:
    - 左侧仍是窄条带中的单色 sprite，右侧仍是带边框的矩形框，二者之间仍由整列分隔符隔开。
    - 每个 mover 颜色仍通过“左右边同色的行标记”与“上下边同色的列标记”定义目标中心点，
      目标集合仍是 row_marks[color] × col_marks[color] 的笛卡尔积。
    - mover 原位擦除，distractor 保持不动，印刻仍按 sprite 的几何中心对齐到目标中心。
  changed_rule:
    - 原题覆盖顺序：按左侧条带里的垂直位置排序，越靠下的 sprite 越早印刻，越靠上的越晚印刻并覆盖。
    - 本变体覆盖顺序：按目标数量 len(row_marks[color]) * len(col_marks[color]) 降序排序；
      目标更多的 sprite 先印刻，目标更少的 sprite 后印刻并覆盖。
    - 若目标数量相同，为避免无定义分支，次级顺序保留原题的“自下而上”稳定排序。

behavior_trace:
  concrete_example:
    - A 色 sprite 在左侧更靠上，且有 2 个行标记 × 2 个列标记，所以有 4 个目标中心。
    - B 色 sprite 在左侧更靠下，且只有 1 个行标记 × 1 个列标记，所以只有 1 个目标中心。
    - 两者在框内至少有一处印刻重叠。
  old_logic:
    - B 因为更靠下而先印刻，A 后印刻，所以重叠像素最终显示 A。
  new_logic:
    - A 因为目标数更多而先印刻，B 后印刻，所以同一重叠像素最终显示 B。
  visible_difference:
    - 输入里可直接看到 A 的边框标记更多；输出里重叠区域却由“目标少”的 B 压在最上层。
    - 因为生成器强制制造“目标数排序”与“上下位置排序”相反的样例，所以不会退化回原题。

complexity_and_visibility_checks:
  - 目标数量只使用少量离散档位 4 / 2 / 1，对应 2x2 / (1x2 or 2x1) / 1x1，便于人眼数标记。
  - 生成器强制至少两种不同目标数量，且至少一对 mover 在框内发生跨颜色重叠；
    若新旧顺序输出相同则直接丢弃样例。
  - 不使用颜色值本身排序，只使用可见的“该颜色在边框上的匹配标记数量”。
  - 行标记和列标记仍在不同 mover 之间保持唯一，避免谁对应谁产生歧义。
  - 同一 mover 自身的多个目标中心仍保持至少一个 sprite 尺寸的间隔，避免自覆盖让规则难读。

implicit_assumption_audit:
  - 原 solver 中真正绑定旧 D7 规则的只有 mover 排序那一行；标记提取、目标构造、裁剪印刻逻辑都不依赖上下顺序。
  - 但一旦改为“按目标数量排序”，solver 就新增一个假设：每个 mover 的目标数量必须可稳定读取。
    生成器因此显式保证每个 mover 同时拥有行标记和列标记，并使用唯一行/列标记防止计数混淆。
  - 另一个新增假设是：样例里需要出现至少一个“顺序改变会改输出”的重叠区域，否则变体肉眼不可见。
    生成器因此构造一对冲突 mover，并拒绝所有 new_order_output == old_order_output 的输入。

placement_legality:
  - 每个目标中心都必须让对应 sprite 在框内完整放下，所以行列标记都从 sprite 的可居中范围里取。
  - 冲突对只在不同 mover 之间制造邻近中心点；同一 mover 内部仍保持足够间距。
  - 左侧条带仍只做无重叠竖向堆叠，mover 按目标数量从上到下摆放，使旧顺序与新顺序系统性相反。
"""


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites: list[dict[str, object]] = []
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
            sprite_mask = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": sprite_mask,
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

    return {
        "g": g,
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "sprites": sprites,
        "targets": targets,
    }


def _render_from_scene(scene: dict[str, object], *, order_mode: str) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    bg = int(scene["bg"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    targets = scene["targets"]
    sprites = scene["sprites"]

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if order_mode == "count":
        movers.sort(
            key=lambda s: (
                -len(targets[int(s["color"])]),
                -((int(s["r0"]) + int(s["r1"])) / 2.0),
                int(s["c0"]),
            )
        )
    elif order_mode == "bottom":
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

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
    Move sprites from the left strip into the framed area at marker-derived targets.

    Variant rule:
    - For each mover color, compute all target centers from matching row and column edge marks.
    - Erase the original sprite and stamp it at every target center.
    - Coverage priority is by target count: more targets first, fewer targets later and therefore on top.
    - Ties fall back to the original bottom-to-top order for determinism.
    """
    scene = _extract_scene(grid)
    return _render_from_scene(scene, order_mode="count")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int, banned: set[int] | None = None) -> list[int] | None:
    """Pick exactly n values from candidates with pairwise spacing and no banned values."""
    banned = set() if banned is None else set(map(int, banned))
    vals = [int(v) for v in candidates if int(v) not in banned]
    if len(vals) < int(n):
        return None
    random.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return sorted(picked)
    return None


def _complete_marks(
    candidates: list[int],
    anchor: int,
    total_count: int,
    min_spacing: int,
    banned: set[int] | None = None,
) -> list[int] | None:
    """Complete a 1- or 2-mark set that must include anchor."""
    banned = set() if banned is None else set(map(int, banned))
    anchor = int(anchor)
    if anchor in banned or anchor not in set(map(int, candidates)):
        return None
    if int(total_count) == 1:
        return [anchor]

    vals = [int(v) for v in candidates if int(v) not in banned and int(v) != int(anchor)]
    random.shuffle(vals)
    for v in vals:
        if abs(int(v) - int(anchor)) >= int(min_spacing):
            return sorted([int(anchor), int(v)])
    return None


def _count_layout(target_count: int) -> tuple[int, int]:
    if int(target_count) == 1:
        return 1, 1
    if int(target_count) == 2:
        return random.choice([(1, 2), (2, 1)])
    if int(target_count) == 4:
        return 2, 2
    raise ValueError(f"unsupported target_count: {target_count}")


def _fit_candidates(sprite: np.ndarray, ir0: int, ir1: int, ic0: int, ic1: int) -> tuple[list[int], list[int]]:
    sh, sw = sprite.shape
    row_min = int(ir0 + sh // 2)
    row_max = int(ir1 - (sh - 1 - sh // 2))
    col_min = int(ic0 + sw // 2)
    col_max = int(ic1 - (sw - 1 - sw // 2))
    if row_min > row_max or col_min > col_max:
        return [], []
    return list(range(int(row_min), int(row_max) + 1)), list(range(int(col_min), int(col_max) + 1))


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a cropped single-color 8-connected sprite."""
    if max_h < 3 or max_w < 2:
        return None
    sh = min(random.choice([3, 3, 3, 4]), max_h)
    sw = min(random.choice([2, 3, 3, 4, 5]), max_w)
    if sh < 3 or sw < 2:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 40):
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
        sprite[pr - min_r, pc - min_c] = int(color)
    return sprite


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None

    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    n_movers = 3 if H >= 13 and random.random() < 0.4 else 2
    count_plan = [4, 2, 1][:n_movers]
    if n_movers == 2 and random.random() < 0.5:
        count_plan = [4, 1]
    n_distractors = 0 if n_movers == 3 else random.choice([0, 0, 1])

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    reserved_rows = 3 * n_distractors
    remaining_budget = H - reserved_rows
    for i, mc in enumerate(mover_colors):
        remaining_movers = n_movers - i - 1
        max_h = min(4, remaining_budget - 3 * remaining_movers)
        sprite = _make_sprite(int(mc), max_h, sprite_area_w)
        if sprite is None:
            return None
        sprites.append(sprite)
        remaining_budget -= int(sprite.shape[0])

    mover_infos: list[dict[str, object]] = []
    for mc, sp, target_count in zip(mover_colors, sprites, count_plan):
        row_n, col_n = _count_layout(int(target_count))
        row_cands, col_cands = _fit_candidates(sp, ir0, ir1, ic0, ic1)
        if not row_cands or not col_cands:
            return None
        mover_infos.append(
            {
                "color": int(mc),
                "sprite": sp,
                "target_count": int(target_count),
                "row_n": int(row_n),
                "col_n": int(col_n),
                "row_cands": row_cands,
                "col_cands": col_cands,
            }
        )

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    leader = mover_infos[0]
    follower = mover_infos[1]
    leader_h, leader_w = leader["sprite"].shape
    follower_h, follower_w = follower["sprite"].shape
    conflict_found = False

    diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(diagonals)
    leader_rows_shuffled = list(leader["row_cands"])
    leader_cols_shuffled = list(leader["col_cands"])
    random.shuffle(leader_rows_shuffled)
    random.shuffle(leader_cols_shuffled)

    for dr, dc in diagonals:
        if conflict_found:
            break
        for ra in leader_rows_shuffled:
            rb = int(ra + dr)
            if rb not in set(follower["row_cands"]):
                continue
            for ca in leader_cols_shuffled:
                cb = int(ca + dc)
                if cb not in set(follower["col_cands"]):
                    continue
                leader_rows = _complete_marks(leader["row_cands"], int(ra), int(leader["row_n"]), int(leader_h), {int(rb)})
                leader_cols = _complete_marks(leader["col_cands"], int(ca), int(leader["col_n"]), int(leader_w), {int(cb)})
                if leader_rows is None or leader_cols is None:
                    continue
                follower_rows = _complete_marks(
                    follower["row_cands"],
                    int(rb),
                    int(follower["row_n"]),
                    int(follower_h),
                    set(map(int, leader_rows)),
                )
                follower_cols = _complete_marks(
                    follower["col_cands"],
                    int(cb),
                    int(follower["col_n"]),
                    int(follower_w),
                    set(map(int, leader_cols)),
                )
                if follower_rows is None or follower_cols is None:
                    continue
                mover_targets[int(leader["color"])] = (leader_rows, leader_cols)
                mover_targets[int(follower["color"])] = (follower_rows, follower_cols)
                used_rows.update(map(int, leader_rows + follower_rows))
                used_cols.update(map(int, leader_cols + follower_cols))
                conflict_found = True
                break

    if not conflict_found:
        return None

    for info in mover_infos[2:]:
        sp = info["sprite"]
        sh, sw = sp.shape
        rows = _pick_spaced(list(info["row_cands"]), int(info["row_n"]), int(sh), used_rows)
        cols = _pick_spaced(list(info["col_cands"]), int(info["col_n"]), int(sw), used_cols)
        if rows is None or cols is None:
            return None
        mover_targets[int(info["color"])] = (rows, cols)
        used_rows.update(map(int, rows))
        used_cols.update(map(int, cols))

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in set(map(int, rmarks)):
                val = int(mc)
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(int(info["color"]), info["sprite"], False) for info in mover_infos]
    for dc_color in distractor_colors:
        max_h = min(4, H - cur_row)
        ds = _make_sprite(int(dc_color), max_h, sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc_color), ds, True))

    for color, sp, _ in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row) + int(dr), int(c_off) + int(dc)] = int(color)
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    scene = _extract_scene(inp)
    planned_counts = {int(info["color"]): int(info["target_count"]) for info in mover_infos}
    for color, expected in planned_counts.items():
        if len(scene["targets"].get(int(color), [])) != int(expected):
            return None

    try:
        new_output = _render_from_scene(scene, order_mode="count")
        old_output = _render_from_scene(scene, order_mode="bottom")
    except Exception:
        return None

    if np.array_equal(inp, new_output):
        return None
    if np.array_equal(new_output, old_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
