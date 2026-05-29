# Combo variant of 247ef758 (index 27)
# combo_spec: D4/1+D12/0+D13/0
# Name: 旋转+左上对齐+环绕溢出

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 骨架仍是: 左侧精灵条带 | 单色分隔列 | 右侧整高矩形框。颜色匹配规则不变: 同色左右边给 row marks，
#   同色上下边给 col marks，笛卡尔积得到目标点。
# - 融合后的印刻链必须是串行的三步，而不是三个互不相关的开关:
#   1. 先看目标点离哪一条框边最近，得到旋转方向:
#      top->0, left->1, bottom->2, right->3（按 np.rot90 的 k）。
#   2. 再把“旋转后的 bbox 左上角”直接放到目标点 `(tr, tc)`。
#   3. 若像素落到 interior 外，不裁剪，而是按 interior 的高宽取模环绕到对侧。
# - 具体小例子: 某个 3x2 非对称 L 精灵，在原题里若目标是 `(6, 12)`，会以中心对齐、原朝向、越界裁剪。
#   在本融合变体里，若 `(6,12)` 更靠右边，则先旋成 right 对应的 270°，再让旋转后 bbox 左上角落在 `(6,12)`，
#   若新宽度把像素推到 `ic1+1` 之外，则这些像素从 `ic0` 重新进入。差异因此同时体现在形状、锚点和边界行为。
# - 一个关键协同点是 D12 与 D13 并不独立: 左上锚点意味着越界只会向下/向右发生，而 D4 的旋转又会交换 h/w，
#   直接改变“会不会在 bottom/right 方向溢出”。所以 wrap 是否发生本身也受旋转控制。
#
# b) 复杂度校准
# - 仍只保留 2-3 个 mover 和 0-1 个 distractor，不引入额外对象类型，避免颜色归属变得模糊。
# - mover 精灵全部使用四向旋转可区分的非对称模板，保证 D4 肉眼可见。
# - 至少一个 special mover 被强制拿到:
#   - 至少两个目标点，且这些目标点的最近边方向不同；
#   - 至少一个目标点发生 wrap；
#   这样三条变化链都会真实触发，而不是只在代码里存在。
#
# c) 隐式假设审查
# - 原题 `cr/cc = bbox center` 的中心锚点前提被 D12 破坏，必须改为“目标点就是旋转后 bbox 左上角”。
# - 原题默认所有副本与源精灵同朝向；D4 破坏该前提，必须先对每个目标单独旋转，再印刻。
# - 原题 `if ir0 <= rr <= ir1 ...` 默认越界即丢弃；D13 破坏该前提，必须改为相对 interior 原点取模。
# - 三项叠加后新增假设:
#   - 目标点的最近边若出现平局，solver 必须有确定 tie-break；这里固定优先级 top/left/bottom/right。
#   - 生成器需要主动规避平局，否则人类难以从少量样例推断 tie-break。
#   - 由于左上锚点只可能向右/下溢出，生成器必须把至少一个目标放到 bottom/right 附近，否则 D13 会退化。
#
# d) 放置合法性
# - 输入阶段仍保证左侧精灵彼此不重叠、不越过 divider。
# - 不同 mover 不共享 row mark 或 col mark，避免目标集合归属歧义。
# - 生成器允许输出时 wrap，但输入中的 frame、markers、sprites 本身都完全合法。
#
# e) 视觉可观察性
# - special mover 强制出现两个不同方向的副本，且至少一个副本在 right/bottom 侧环绕，解题者能从样例中看出:
#   “目标点附近的形状会随边方向转动，并且不是中心落点，而是从交点往右下展开，越界还能从另一侧冒出来”。
# - 不使用颜色值大小、背景特权等不可见因素做决策；方向完全由空间距离决定。
#
# f) 可视化整洁度
# - special mover 放在左条带最上方，使其在覆盖顺序中最后印刻，减少被别的 mover 遮住。
# - 其余 mover 优先选择不 wrap 或少 wrap 的 marks，避免整个 interior 到处断裂回卷而显得杂乱。
# - mark 间距仍按精灵最大跨度控制，减少同色副本相互重叠。
#
# g) 变化的可推断性
# - 颜色职责不变: 颜色只负责“哪只左侧精灵对应哪组边标”。
# - 旋转、锚点、wrap 都由目标点的空间位置决定，且在同一颜色的多个副本中会重复出现，便于归纳。
#
# h) main() 的新隐式假设与 generator 保证
# - `main()` 现在假设 mover 精灵是 8 连通单色组件，生成器保持这一点。
# - `main()` 假设至少存在一个最大的不接边背景封闭区作为 interior，生成器仍使用原题的整高矩形框布局。
# - `main()` 在 tie 时按 top/left/bottom/right 兜底；生成器通过 `_nearest_side_code(..., strict=True)` 只采样无平局目标。
# - `main()` 不要求目标能完整容纳精灵；生成器正是利用 bottom/right 靠边 marks 触发 wrap。


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does NOT touch the grid border.
    """
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
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    sigs: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for krot in range(4):
        rot = _crop_bool(np.rot90(mask.astype(bool), int(krot)))
        pts = tuple(tuple(map(int, p)) for p in np.argwhere(rot))
        sigs.add(((int(rot.shape[0]), int(rot.shape[1])), pts))
    return sigs


def _make_asymmetric_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Build a single-color sprite whose four rotations are visually distinguishable.
    """
    templates = [
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    ]

    candidates: list[np.ndarray] = []
    for tpl in templates:
        bool_tpl = tpl.astype(bool)
        if len(_rotation_signatures(bool_tpl)) != 4:
            continue
        for krot in range(4):
            rot = _crop_bool(np.rot90(bool_tpl, int(krot)))
            if int(rot.shape[0]) <= int(max_h) and int(rot.shape[1]) <= int(max_w):
                arr = np.zeros(rot.shape, dtype=int)
                arr[rot] = int(color)
                candidates.append(arr)
    if not candidates:
        return None
    return np.array(random.choice(candidates), dtype=int)


def _nearest_side_code(
    tr: int,
    tc: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
    *,
    strict: bool,
) -> int | None:
    dists = {
        0: int(tr - fr0),   # top
        1: int(tc - fc0),   # left
        2: int(fr1 - tr),   # bottom
        3: int(fc1 - tc),   # right
    }
    best = min(dists.values())
    winners = [int(code) for code, dist in dists.items() if int(dist) == int(best)]
    if strict and len(winners) != 1:
        return None
    return int(min(winners))


def _extract_scene(grid: np.ndarray):
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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": np.array(m, dtype=bool),
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

    return g, bg, (ir0, ic0, ir1, ic1), (fr0, fc0, fr1, fc1), sprites, targets


def _stamp_cells(
    mask: np.ndarray,
    target: tuple[int, int],
    interior: tuple[int, int, int, int],
    *,
    anchor_mode: str,
    wrap: bool,
) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = map(int, interior)
    inner_h = int(ir1 - ir0 + 1)
    inner_w = int(ic1 - ic0 + 1)
    tr, tc = map(int, target)
    h, w = map(int, mask.shape)

    if anchor_mode == "center":
        anchor_r = int((h - 1) // 2)
        anchor_c = int((w - 1) // 2)
    elif anchor_mode == "topleft":
        anchor_r = 0
        anchor_c = 0
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    base_r = int(tr - anchor_r)
    base_c = int(tc - anchor_c)
    cells: list[tuple[int, int]] = []

    for rr, cc in np.argwhere(mask):
        raw_r = int(base_r + int(rr))
        raw_c = int(base_c + int(cc))
        if wrap:
            pr = int(ir0 + ((raw_r - ir0) % inner_h))
            pc = int(ic0 + ((raw_c - ic0) % inner_w))
            cells.append((pr, pc))
        elif int(ir0) <= raw_r <= int(ir1) and int(ic0) <= raw_c <= int(ic1):
            cells.append((raw_r, raw_c))

    return cells


def _target_stats(
    base_mask: np.ndarray,
    targets: list[tuple[int, int]],
    interior: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
    *,
    rotate: bool,
    anchor_mode: str,
) -> dict[str, int] | None:
    ir0, ic0, ir1, ic1 = map(int, interior)
    fr0, fc0, fr1, fc1 = map(int, frame)
    inner_h = int(ir1 - ir0 + 1)
    inner_w = int(ic1 - ic0 + 1)

    rotations: set[int] = set()
    wrap_targets = 0
    wrap_pixels = 0

    for tr, tc in targets:
        code = 0
        if rotate:
            maybe_code = _nearest_side_code(int(tr), int(tc), int(fr0), int(fc0), int(fr1), int(fc1), strict=True)
            if maybe_code is None:
                return None
            code = int(maybe_code)
        rotations.add(int(code))
        rot_mask = _crop_bool(np.rot90(base_mask, int(code)))

        h, w = map(int, rot_mask.shape)
        if anchor_mode == "center":
            anchor_r = int((h - 1) // 2)
            anchor_c = int((w - 1) // 2)
        else:
            anchor_r = 0
            anchor_c = 0
        base_r = int(tr - anchor_r)
        base_c = int(tc - anchor_c)

        this_wrap = False
        for rr, cc in np.argwhere(rot_mask):
            raw_r = int(base_r + int(rr))
            raw_c = int(base_c + int(cc))
            if not (int(ir0) <= raw_r <= int(ir1) and int(ic0) <= raw_c <= int(ic1)):
                wrap_pixels += 1
                this_wrap = True
        if this_wrap:
            wrap_targets += 1

    return {
        "distinct_rotations": int(len(rotations)),
        "rotated_targets": int(sum(1 for r in rotations if int(r) != 0)),
        "wrap_targets": int(wrap_targets),
        "wrap_pixels": int(wrap_pixels),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
    }


def _solve(
    grid: np.ndarray,
    *,
    rotate: bool,
    anchor_mode: str,
    wrap: bool,
) -> np.ndarray:
    g, bg, interior, frame, sprites, targets = _extract_scene(grid)
    _fr0, _fc0, _fr1, _fc1 = frame
    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        base_mask = np.array(s["mask"], dtype=bool)
        for tr, tc in targets[int(col)]:
            code = 0
            if rotate:
                maybe_code = _nearest_side_code(
                    int(tr),
                    int(tc),
                    int(_fr0),
                    int(_fc0),
                    int(_fr1),
                    int(_fc1),
                    strict=False,
                )
                assert maybe_code is not None
                code = int(maybe_code)
            rot_mask = _crop_bool(np.rot90(base_mask, int(code)))
            for rr, cc in _stamp_cells(
                rot_mask,
                (int(tr), int(tc)),
                interior,
                anchor_mode=str(anchor_mode),
                wrap=bool(wrap),
            ):
                out[int(rr), int(cc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Targets are still row_marks x col_marks.
    - At each target, rotate by nearest frame-side direction.
    - Use the rotated sprite's bbox top-left corner as the anchor.
    - Wrap any bottom/right overflow toroidally across the frame interior.
    - Lower input sprites are painted first; upper sprites cover them.
    """
    return _solve(grid, rotate=True, anchor_mode="topleft", wrap=True)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    if n == 1:
        return [int(random.choice(candidates))]
    pool = list(map(int, candidates))
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return picked
    return picked if picked else None


def _spaced_mark_options(candidates: list[int], min_spacing: int) -> list[list[int]]:
    opts: list[list[int]] = []
    vals = sorted(set(map(int, candidates)))
    for n in (1, 2):
        if len(vals) < int(n):
            continue
        for combo in combinations(vals, n):
            if all(abs(int(combo[j]) - int(combo[i])) >= int(min_spacing) for i in range(n) for j in range(i + 1, n)):
                opts.append([int(v) for v in combo])
    random.shuffle(opts)
    return opts


def _pick_special_marks(
    base_mask: np.ndarray,
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    interior: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
    *,
    mode: str,
) -> tuple[list[int], list[int]] | None:
    for _ in range(220):
        if mode == "horizontal":
            rows = _pick_spaced(list(row_candidates), 1, int(span))
            cols = _pick_spaced(list(col_candidates), 2, int(span))
            wanted = {1, 3}
        elif mode == "vertical":
            rows = _pick_spaced(list(row_candidates), 2, int(span))
            cols = _pick_spaced(list(col_candidates), 1, int(span))
            wanted = {0, 2}
        else:
            return None
        if not rows or not cols:
            continue
        targets = [(int(r), int(c)) for r in rows for c in cols]
        stats = _target_stats(base_mask, targets, interior, frame, rotate=True, anchor_mode="topleft")
        if stats is None:
            continue
        rotations = {
            _nearest_side_code(int(r), int(c), *map(int, frame), strict=True)
            for r in rows
            for c in cols
        }
        if None in rotations:
            continue
        if set(map(int, rotations)) != set(map(int, wanted)):
            continue
        if int(stats["wrap_targets"]) <= 0:
            continue
        return sorted(map(int, rows)), sorted(map(int, cols))
    return None


def _pick_generic_marks(
    base_mask: np.ndarray,
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    interior: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
) -> tuple[list[int], list[int]] | None:
    fallback: tuple[list[int], list[int], int, int] | None = None
    weighted_options = [(1, 1), (1, 1), (2, 1), (1, 2)]
    random.shuffle(weighted_options)

    for row_n, col_n in weighted_options:
        for _ in range(100):
            rows = _pick_spaced(list(row_candidates), int(row_n), int(span))
            cols = _pick_spaced(list(col_candidates), int(col_n), int(span))
            if not rows or not cols or len(rows) != int(row_n) or len(cols) != int(col_n):
                continue
            targets = [(int(r), int(c)) for r in rows for c in cols]
            stats = _target_stats(base_mask, targets, interior, frame, rotate=True, anchor_mode="topleft")
            if stats is None:
                continue
            if int(stats["wrap_targets"]) == 0:
                return sorted(map(int, rows)), sorted(map(int, cols))
            rec = (sorted(map(int, rows)), sorted(map(int, cols)), int(stats["wrap_targets"]), int(stats["wrap_pixels"]))
            if fallback is None or (int(rec[2]), int(rec[3])) < (int(fallback[2]), int(fallback[3])):
                fallback = rec

    if fallback is None:
        return None
    rows, cols, _wt, _wp = fallback
    return rows, cols


def _assign_targets(
    sprites: list[np.ndarray],
    mover_colors: list[int],
    interior: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
) -> dict[int, tuple[list[int], list[int]]] | None:
    ir0, ic0, ir1, ic1 = map(int, interior)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    special_modes = ["horizontal", "vertical"]
    random.shuffle(special_modes)

    for idx, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        span = int(max(sp.shape))
        row_candidates = [int(r) for r in range(int(ir0), int(ir1) + 1) if int(r) not in occupied_rows]
        col_candidates = [int(c) for c in range(int(ic0), int(ic1) + 1) if int(c) not in occupied_cols]
        if not row_candidates or not col_candidates:
            return None

        if idx == 0:
            picked = None
            for mode in special_modes:
                picked = _pick_special_marks(
                    sp != 0,
                    row_candidates,
                    col_candidates,
                    int(span),
                    interior,
                    frame,
                    mode=str(mode),
                )
                if picked is not None:
                    break
            if picked is None:
                return None
        else:
            picked = _pick_generic_marks(
                sp != 0,
                row_candidates,
                col_candidates,
                int(span),
                interior,
                frame,
            )
            if picked is None:
                return None

        rows, cols = picked
        mover_targets[int(mc)] = (list(map(int, rows)), list(map(int, cols)))
        occupied_rows.update(map(int, rows))
        occupied_cols.update(map(int, cols))

    special_rows, special_cols = mover_targets[int(mover_colors[0])]
    special_targets = [(int(r), int(c)) for r in special_rows for c in special_cols]
    stats = _target_stats(sprites[0] != 0, special_targets, interior, frame, rotate=True, anchor_mode="topleft")
    if stats is None or int(stats["distinct_rotations"]) < 2 or int(stats["wrap_targets"]) <= 0:
        return None

    return mover_targets


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a generated input truly exercises all three fused controls.
    """
    g, _bg, interior, frame, sprites, targets = _extract_scene(grid)
    distinct_rotations: set[int] = set()
    total_targets = 0
    wrap_targets = 0
    wrap_pixels = 0

    for s in sprites:
        col = int(s["color"])
        if col not in targets:
            continue
        total_targets += int(len(targets[col]))
        stats = _target_stats(s["mask"], targets[col], interior, frame, rotate=True, anchor_mode="topleft")
        if stats is None:
            continue
        wrap_targets += int(stats["wrap_targets"])
        wrap_pixels += int(stats["wrap_pixels"])
        for tr, tc in targets[col]:
            code = _nearest_side_code(int(tr), int(tc), *map(int, frame), strict=True)
            if code is not None:
                distinct_rotations.add(int(code))

    fused = _solve(g, rotate=True, anchor_mode="topleft", wrap=True)
    rot_center_wrap = _solve(g, rotate=True, anchor_mode="center", wrap=True)
    rot_topleft_clip = _solve(g, rotate=True, anchor_mode="topleft", wrap=False)
    norot_topleft_wrap = _solve(g, rotate=False, anchor_mode="topleft", wrap=True)
    original = _solve(g, rotate=False, anchor_mode="center", wrap=False)

    return {
        "total_targets": int(total_targets),
        "distinct_rotations": int(len(distinct_rotations)),
        "wrap_targets": int(wrap_targets),
        "wrap_pixels": int(wrap_pixels),
        "diff_vs_rot_center_wrap": int(np.count_nonzero(fused != rot_center_wrap)),
        "diff_vs_rot_topleft_clip": int(np.count_nonzero(fused != rot_topleft_clip)),
        "diff_vs_norot_topleft_wrap": int(np.count_nonzero(fused != norot_topleft_wrap)),
        "diff_vs_original": int(np.count_nonzero(fused != original)),
    }


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

    div_col = int(sprite_area_w)
    fc0, fc1 = int(sprite_area_w + 1), int(W - 1)
    ir0, ic0 = 1, int(fc0 + 1)
    ir1, ic1 = int(H - 2), int(fc1 - 1)
    interior = (int(ir0), int(ic0), int(ir1), int(ic1))
    frame = (0, int(fc0), int(H - 1), int(fc1))

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_asymmetric_sprite(int(mc), 5, int(sprite_area_w))
        if sp is None:
            return None
        sprites.append(np.array(sp, dtype=int))

    if not any(int(sp.shape[0]) > 1 and int(sp.shape[1]) > 1 for sp in sprites):
        return None

    mover_targets = _assign_targets(sprites, mover_colors, interior, frame)
    if mover_targets is None:
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[H - 1, int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if int(r) in set(map(int, rmarks)):
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(int(mc), np.array(sp, dtype=int)) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_asymmetric_sprite(int(dc_color), min(4, H - cur_row), int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), np.array(ds, dtype=int)))

    for color, sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > int(sprite_area_w) or cur_row + sh > int(H):
            return None
        c_off = int(random.randint(0, int(sprite_area_w - sw)))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(color)
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        fused = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, fused):
        return None

    metrics = analyze_variant(inp)
    if int(metrics["distinct_rotations"]) < 2:
        return None
    if int(metrics["wrap_targets"]) <= 0 or int(metrics["wrap_pixels"]) <= 0:
        return None
    if int(metrics["diff_vs_rot_center_wrap"]) < 2:
        return None
    if int(metrics["diff_vs_rot_topleft_clip"]) < 2:
        return None
    if int(metrics["diff_vs_norot_topleft_wrap"]) < 2:
        return None
    if int(metrics["diff_vs_original"]) < 3:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
