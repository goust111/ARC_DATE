# Combo variant of 247ef758 (index 25)
# combo_spec: D3/1+D7/3
# Name: 多框架+标记数量排序

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask

"""
Design analysis
===============

fused_rule_summary:
  - 布局从“左条带 + 单框”改成“左条带 + 2 个独立框架”；两个框架边框基色不同。
  - 每个 sprite 仍按颜色去找“左右边同色的行标记 + 上下边同色的列标记”，
    但现在必须先判断它属于哪个框架，再在该框架内部做 row x col 的笛卡尔积印刻。
  - 覆盖顺序不再看左侧上下位置，而看该颜色拥有的目标数量：
      len(row_marks[color]) * len(col_marks[color])
    目标多的先印刻，目标少的后印刻并覆盖；同数时再退回原题的自下而上稳定次序。

behavior_trace_example:
  - 右侧有两个 7x7 独立框架。上框同时含 red 与 blue 的边缘标记，下框含 green 的标记。
  - red 在上框有 2 个行标记 x 2 个列标记，共 4 个目标；blue 只有 1x1，共 1 个目标。
    red 与 blue 在上框至少有一处印刻重叠。green 在另一框有 1x2，共 2 个目标。
  - 原题/仅 D3 逻辑若沿用“看左侧上下位置”，会让条带中更靠上的 sprite 更晚盖住下面的。
  - 融合后必须先做“颜色 -> 框架”的路由，再按目标数 4 > 2 > 1 排序印刻。
    因此同一上框中的 red 会先印，blue 后印并压住 red；这正是 D3 与 D7 两项改动的协同点。

complexity_calibration:
  - 生成器固定使用 2 个独立框架。原因是该组合只有 3 个 mover 时，若扩到 3 框就容易出现空框，
    会削弱“排序改变输出”的可观察性；2 框已满足“2-3 个独立框架”的规格下限。
  - 目标数量只使用 4 / 2 / 1 三档，分别对应 2x2 / 1x2 或 2x1 / 1x1，肉眼可直接数边标记。
  - 至少两种颜色被路由到同一框架，并强制它们在该框内部产生跨颜色重叠；
    否则多框架成立但覆盖顺序没有可见作用，会退化。

implicit_assumption_audit:
  - 原 solver 的“找最大的唯一 enclosed bg 组件”失效，因为现在有多个独立框架。
    新 solver 改为枚举所有不接触边界的 bg 矩形内腔。
  - 原 solver 用单一 interior bbox 裁剪印刻；融合后每个 target 都要携带自己的框内边界。
  - 原 solver 默认所有 mover 都去同一个框；融合后要逐框读标记，再按颜色收集到对应目标集。
  - 新 solver 新增的隐式假设：
      * 每个真实框架都有完整非背景边框，四角同基色；
      * divider 是唯一整列满高单色非背景列；
      * 每个 mover 只在一个框架中拥有完整的行/列标记；
      * 至少一对同框 mover 的目标数不同且印刻有重叠。
    这些都由生成器显式保证。

placement_legality:
  - 两个框架放在 2x2 槽位中的两个 7x7 位置，彼此之间至少隔一行/一列背景。
  - 框内可用 interior 固定为 5x5；sprite 高宽限制在 4 以内，所有目标中心都从“完整放下 sprite”
    的可行中心集合里采样。
  - 同一框内不同颜色不共享行标记或列标记，避免边缘像素被二次覆写造成歧义。
  - 左条带中的 sprite 按目标数量从上到下摆放，从而系统性地与“按目标数排序”形成对照。

visual_observability_and_cleanliness:
  - 多框架是显式可见结构差异：右侧有两个相互分离的盒子，且基色不同。
  - 计数规则只依赖边缘标记的可见个数，不依赖颜色值、生成时顺序等不可见信息。
  - 只有一个框架承载跨颜色冲突，另一个框架只放单色 mover，避免所有区域都层层覆盖而显得杂乱。
"""


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Detect all enclosed rectangular frames via their bg interiors."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    lab, n = label_mask(g == int(bg), connectivity=4)
    frames: list[dict[str, int]] = []

    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        ir0, ic0 = coords.min(axis=0)
        ir1, ic1 = coords.max(axis=0)
        ir0, ic0, ir1, ic1 = map(int, (ir0, ic0, ir1, ic1))
        fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if fr0 < 0 or fc0 < 0 or fr1 >= H or fc1 >= W:
            continue
        if not bool(np.all(g[ir0 : ir1 + 1, ic0 : ic1 + 1] == int(bg))):
            continue

        top = g[fr0, fc0 : fc1 + 1]
        bottom = g[fr1, fc0 : fc1 + 1]
        left = g[fr0 : fr1 + 1, fc0]
        right = g[fr0 : fr1 + 1, fc1]
        ring = np.concatenate([top, bottom, left, right])
        if bool((ring == int(bg)).any()):
            continue

        corners = [int(g[fr0, fc0]), int(g[fr0, fc1]), int(g[fr1, fc0]), int(g[fr1, fc1])]
        if len(set(corners)) != 1:
            continue

        frames.append(
            {
                "fr0": int(fr0),
                "fc0": int(fc0),
                "fr1": int(fr1),
                "fc1": int(fc1),
                "ir0": int(ir0),
                "ic0": int(ic0),
                "ir1": int(ir1),
                "ic1": int(ic1),
                "base": int(corners[0]),
            }
        )

    frames.sort(key=lambda d: (int(d["fr0"]), int(d["fc0"])))
    return frames


def _find_divider_mask(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    """Find the full-height uniform non-bg divider column."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    divider_mask = np.zeros((H, W), dtype=bool)
    divider_cols: list[int] = []
    for c in range(W):
        colv = g[:, int(c)]
        if bool((colv != int(bg)).all()) and len(set(map(int, colv))) == 1:
            divider_mask[:, int(c)] = True
            divider_cols.append(int(c))
    assert len(divider_cols) >= 1
    return divider_mask, int(min(divider_cols))


def _extract_sprites(grid: np.ndarray, bg: int, divider_mask: np.ndarray, divider_col: int) -> list[dict[str, object]]:
    """Extract 8-connected single-color sprites from the left strip."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    left_cols = np.arange(W)[None, :] < int(divider_col)
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

    sprites.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    return sprites


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _find_frames(g, bg)
    assert len(frames) >= 2

    divider_mask, divider_col = _find_divider_mask(g, bg)
    sprites = _extract_sprites(g, bg, divider_mask, divider_col)
    move_colors = {int(s["color"]) for s in sprites}

    targets: dict[int, list[tuple[int, int, int, int, int, int]]] = defaultdict(list)
    row_marks_by_color: dict[int, list[int]] = defaultdict(list)
    col_marks_by_color: dict[int, list[int]] = defaultdict(list)

    for frame_idx, frame in enumerate(frames):
        ir0, ic0, ir1, ic1 = (int(frame[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
        fr0, fc0, fr1, fc1 = (int(frame[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        local_rows: dict[int, list[int]] = defaultdict(list)
        local_cols: dict[int, list[int]] = defaultdict(list)

        for r in range(ir0, ir1 + 1):
            cl = int(g[int(r), int(fc0)])
            cr = int(g[int(r), int(fc1)])
            if cl == cr and cl != int(bg) and cl in move_colors:
                local_rows[int(cl)].append(int(r))

        for c in range(ic0, ic1 + 1):
            ct = int(g[int(fr0), int(c)])
            cb = int(g[int(fr1), int(c)])
            if ct == cb and ct != int(bg) and ct in move_colors:
                local_cols[int(ct)].append(int(c))

        for col in sorted(set(local_rows.keys()) & set(local_cols.keys())):
            row_marks_by_color[int(col)] = list(map(int, local_rows[int(col)]))
            col_marks_by_color[int(col)] = list(map(int, local_cols[int(col)]))
            for tr in local_rows[int(col)]:
                for tc in local_cols[int(col)]:
                    targets[int(col)].append((int(tr), int(tc), ir0, ic0, ir1, ic1))

    target_counts = {int(col): int(len(pts)) for col, pts in targets.items()}

    return {
        "g": g,
        "bg": int(bg),
        "frames": frames,
        "divider_col": int(divider_col),
        "sprites": sprites,
        "targets": targets,
        "target_counts": target_counts,
        "row_marks": row_marks_by_color,
        "col_marks": col_marks_by_color,
    }


def _render_scene(scene: dict[str, object], *, order_mode: str) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    bg = int(scene["bg"])
    targets = scene["targets"]
    target_counts = scene["target_counts"]
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
                -int(target_counts[int(s["color"])]),
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

        for tr, tc, ir0, ic0, ir1, ic1 in targets[int(col)]:
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
    Move left-strip sprites into their matching frames, then stamp by target-count priority.

    Rule:
    - Detect every independent enclosed frame on the right.
    - Within each frame, read matching opposite-edge markers to derive target centers.
    - Route each sprite color to the frame(s) carrying its row+col markers.
    - Erase moved sprites from the left strip and stamp them centered at every target.
    - Sort movers by number of targets descending; fewer-target sprites stamp later and cover more-target ones.
      Ties use the original bottom-to-top order for determinism.
    """
    scene = _extract_scene(grid)
    return _render_scene(scene, order_mode="count")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int, banned: set[int] | None = None) -> list[int] | None:
    """Pick exactly n values with pairwise spacing and optional banned set."""
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
    """Build a 1- or 2-mark set that must include anchor."""
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
            return sorted([anchor, int(v)])
    return None


def _count_layout(target_count: int) -> tuple[int, int]:
    if int(target_count) == 1:
        return 1, 1
    if int(target_count) == 2:
        return random.choice([(1, 2), (2, 1)])
    if int(target_count) == 4:
        return 2, 2
    raise ValueError(f"unsupported target_count: {target_count}")


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a cropped 8-connected single-color sprite that fits a 5x5 interior."""
    if max_h < 2 or max_w < 2:
        return None
    sh = min(random.choice([2, 3, 3, 4]), max_h)
    sw = min(random.choice([2, 3, 3, 4]), max_w)
    if sh < 2 or sw < 2:
        return None

    n_pixels = random.randint(3, min(7, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 40):
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


def _fit_candidates(sprite: np.ndarray, ir0: int, ir1: int, ic0: int, ic1: int) -> tuple[list[int], list[int]]:
    sh, sw = sprite.shape
    row_min = int(ir0 + sh // 2)
    row_max = int(ir1 - (sh - 1 - sh // 2))
    col_min = int(ic0 + sw // 2)
    col_max = int(ic1 - (sw - 1 - sw // 2))
    if row_min > row_max or col_min > col_max:
        return [], []
    return list(range(int(row_min), int(row_max) + 1)), list(range(int(col_min), int(col_max) + 1))


def _frame_slots(frame_start_col: int) -> list[tuple[int, int]]:
    return [
        (0, int(frame_start_col)),
        (0, int(frame_start_col + 8)),
        (8, int(frame_start_col)),
        (8, int(frame_start_col + 8)),
    ]


def _stamp_pixels(sprite: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    sh, sw = sprite.shape
    cr = int(sh // 2)
    cc = int(sw // 2)
    tr, tc = map(int, center)
    pixels: set[tuple[int, int]] = set()
    for rr, cc2 in np.argwhere(sprite):
        pixels.add((int(tr - cr + rr), int(tc - cc + cc2)))
    return pixels


def _all_stamped_pixels(sprite: np.ndarray, rows: list[int], cols: list[int]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for r in rows:
        for c in cols:
            pixels |= _stamp_pixels(sprite, (int(r), int(c)))
    return pixels


def _plan_conflict_pair(
    frame: dict[str, int],
    leader: dict[str, object],
    follower: dict[str, object],
) -> tuple[list[int], list[int], list[int], list[int]] | None:
    """Choose marks for two movers in the same frame so their stamps overlap."""
    ir0, ic0, ir1, ic1 = (int(frame[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
    leader_sprite = np.array(leader["sprite"], dtype=int)
    follower_sprite = np.array(follower["sprite"], dtype=int)
    leader_rows_cand, leader_cols_cand = _fit_candidates(leader_sprite, ir0, ir1, ic0, ic1)
    follower_rows_cand, follower_cols_cand = _fit_candidates(follower_sprite, ir0, ir1, ic0, ic1)
    if not leader_rows_cand or not leader_cols_cand or not follower_rows_cand or not follower_cols_cand:
        return None

    leader_row_n, leader_col_n = _count_layout(int(leader["target_count"]))
    follower_row_n, follower_col_n = _count_layout(int(follower["target_count"]))
    leader_h, leader_w = leader_sprite.shape
    follower_h, follower_w = follower_sprite.shape

    leader_rows = list(leader_rows_cand)
    leader_cols = list(leader_cols_cand)
    random.shuffle(leader_rows)
    random.shuffle(leader_cols)
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
    random.shuffle(offsets)

    follower_row_set = set(map(int, follower_rows_cand))
    follower_col_set = set(map(int, follower_cols_cand))

    for ra in leader_rows:
        for ca in leader_cols:
            leader_rmarks = _complete_marks(leader_rows_cand, int(ra), int(leader_row_n), int(leader_h))
            leader_cmarks = _complete_marks(leader_cols_cand, int(ca), int(leader_col_n), int(leader_w))
            if leader_rmarks is None or leader_cmarks is None:
                continue
            for dr, dc in offsets:
                rb = int(ra + dr)
                cb = int(ca + dc)
                if rb not in follower_row_set or cb not in follower_col_set:
                    continue
                follower_rmarks = _complete_marks(
                    follower_rows_cand,
                    int(rb),
                    int(follower_row_n),
                    int(follower_h),
                    set(map(int, leader_rmarks)),
                )
                follower_cmarks = _complete_marks(
                    follower_cols_cand,
                    int(cb),
                    int(follower_col_n),
                    int(follower_w),
                    set(map(int, leader_cmarks)),
                )
                if follower_rmarks is None or follower_cmarks is None:
                    continue
                leader_pixels = _all_stamped_pixels(leader_sprite, leader_rmarks, leader_cmarks)
                follower_pixels = _all_stamped_pixels(follower_sprite, follower_rmarks, follower_cmarks)
                if leader_pixels & follower_pixels:
                    return leader_rmarks, leader_cmarks, follower_rmarks, follower_cmarks

    return None


def _choose_single_frame_marks(frame: dict[str, int], sprite: np.ndarray, target_count: int) -> tuple[list[int], list[int]] | None:
    ir0, ic0, ir1, ic1 = (int(frame[k]) for k in ("ir0", "ic0", "ir1", "ic1"))
    row_cands, col_cands = _fit_candidates(sprite, ir0, ir1, ic0, ic1)
    if not row_cands or not col_cands:
        return None
    row_n, col_n = _count_layout(int(target_count))
    rows = _pick_spaced(row_cands, int(row_n), max(2, int(sprite.shape[0])))
    cols = _pick_spaced(col_cands, int(col_n), max(2, int(sprite.shape[1])))
    if rows is None or cols is None:
        return None
    return rows, cols


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = 15
    sprite_area_w = random.randint(3, 4)
    divider_col = int(sprite_area_w)
    frame_start_col = int(divider_col + 1)
    W = int(frame_start_col + 15)
    if not (14 <= W <= 20):
        return None

    n_distractors = random.choice([0, 0, 1])
    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = int(all_colors[idx]); idx += 1
    frame_bases = [int(all_colors[idx]), int(all_colors[idx + 1])]
    idx += 2
    mover_colors = [int(v) for v in all_colors[idx : idx + 3]]
    idx += 3
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    count_plan = [4, 2, 1]
    leader_color, solo_color, follower_color = mover_colors

    mover_sprites: dict[int, np.ndarray] = {}
    for color in mover_colors:
        sprite = _make_sprite(int(color), 4, sprite_area_w)
        if sprite is None or sprite.shape[0] > 4 or sprite.shape[1] > 4:
            return None
        mover_sprites[int(color)] = sprite

    slots = _frame_slots(frame_start_col)
    chosen_slots = random.sample(slots, 2)
    frame_infos = []
    for (fr0, fc0), base in zip(chosen_slots, frame_bases):
        frame_infos.append(
            {
                "fr0": int(fr0),
                "fc0": int(fc0),
                "fr1": int(fr0 + 6),
                "fc1": int(fc0 + 6),
                "ir0": int(fr0 + 1),
                "ic0": int(fc0 + 1),
                "ir1": int(fr0 + 5),
                "ic1": int(fc0 + 5),
                "base": int(base),
            }
        )
    frame_infos.sort(key=lambda d: (int(d["fr0"]), int(d["fc0"])))
    conflict_frame = frame_infos[0]
    solo_frame = frame_infos[1]

    leader_info = {"color": int(leader_color), "sprite": mover_sprites[int(leader_color)], "target_count": 4}
    follower_info = {"color": int(follower_color), "sprite": mover_sprites[int(follower_color)], "target_count": 1}
    planned = _plan_conflict_pair(conflict_frame, leader_info, follower_info)
    if planned is None:
        return None
    leader_rows, leader_cols, follower_rows, follower_cols = planned

    solo_marks = _choose_single_frame_marks(solo_frame, mover_sprites[int(solo_color)], 2)
    if solo_marks is None:
        return None
    solo_rows, solo_cols = solo_marks

    mover_targets = {
        int(leader_color): {
            "frame": conflict_frame,
            "rows": list(map(int, leader_rows)),
            "cols": list(map(int, leader_cols)),
            "count": 4,
        },
        int(follower_color): {
            "frame": conflict_frame,
            "rows": list(map(int, follower_rows)),
            "cols": list(map(int, follower_cols)),
            "count": 1,
        },
        int(solo_color): {
            "frame": solo_frame,
            "rows": list(map(int, solo_rows)),
            "cols": list(map(int, solo_cols)),
            "count": 2,
        },
    }

    inp = np.full((H, W), bg, dtype=int)
    inp[:, divider_col] = int(divider_color)

    for frame in frame_infos:
        fr0, fc0, fr1, fc1 = (int(frame[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        base = int(frame["base"])
        inp[fr0, fc0 : fc1 + 1] = int(base)
        inp[fr1, fc0 : fc1 + 1] = int(base)
        inp[fr0 : fr1 + 1, fc0] = int(base)
        inp[fr0 : fr1 + 1, fc1] = int(base)

    for color, info in mover_targets.items():
        frame = info["frame"]
        fr0, fc0, fr1, fc1 = (int(frame[k]) for k in ("fr0", "fc0", "fr1", "fc1"))
        for r in info["rows"]:
            inp[int(r), int(fc0)] = int(color)
            inp[int(r), int(fc1)] = int(color)
        for c in info["cols"]:
            inp[int(fr0), int(c)] = int(color)
            inp[int(fr1), int(c)] = int(color)

    items: list[tuple[int, np.ndarray, bool, int]] = [
        (int(leader_color), mover_sprites[int(leader_color)], False, 4),
        (int(solo_color), mover_sprites[int(solo_color)], False, 2),
        (int(follower_color), mover_sprites[int(follower_color)], False, 1),
    ]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), 4, sprite_area_w)
        if ds is None:
            return None
        items.append((int(dc_color), ds, True, -1))

    heights = [int(sp.shape[0]) for _, sp, _, _ in items]
    needed = sum(heights)
    extra = H - needed
    if extra < 0:
        return None
    gaps = [0] * (len(items) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    cur_row = int(gaps[0])
    for idx_item, (color, sp, _, _) in enumerate(items):
        sh, sw = map(int, sp.shape)
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row) + int(dr), int(c_off) + int(dc)] = int(color)
        cur_row += int(sh) + int(gaps[idx_item + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        scene = _extract_scene(inp)
    except Exception:
        return None

    if len(scene["frames"]) != 2:
        return None

    expected_counts = {int(color): int(info["count"]) for color, info in mover_targets.items()}
    for color, expected in expected_counts.items():
        if int(scene["target_counts"].get(int(color), 0)) != int(expected):
            return None

    try:
        new_output = _render_scene(scene, order_mode="count")
        old_output = _render_scene(scene, order_mode="bottom")
    except Exception:
        return None

    if np.array_equal(inp, new_output):
        return None
    if np.array_equal(new_output, old_output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
