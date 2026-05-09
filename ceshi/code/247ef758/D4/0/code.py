# Variant of 247ef758
# Dimension: D4 — 变换：印刻与擦除行为 (transform)
# Change: 擦除左侧原始精灵，在目标位置居中印刻精灵图案，像素超出框架内部时裁剪（clipping）... → 印刻时重着色
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 原题中，左侧 mover 精灵按自身颜色匹配框架边缘标记，并在目标交点处以“原颜色”印刻。
#     - 本变体保留“颜色匹配决定目标位置、擦除左侧原件、居中印刻、底部优先覆盖”的结构，
#       但把“印刻颜色”改为框架基色 frame_base。
#     - 因此：输入里 mover 仍然是不同颜色；输出里框架内部出现的复制体全部变成同一个 frame_base。
#   concrete_trace:
#     - 例：红色 sprite 由左右边缘的红色行标记和上下边缘的红色列标记决定 2 个目标点。
#     - 原逻辑：在这 2 个目标点印出红色 blob。
#     - 新逻辑：仍在这 2 个目标点印出同样形状的 blob，但 blob 的所有像素都改成 frame_base。
#     - 所以真正变化的流程环节只有“写像素值”这一环；目标位置推断完全不变。
#   complexity_calibration:
#     - 颜色变换只采用一种全局规则：所有 mover 都重着色为 frame_base。
#     - 不引入按 mover 颜色分支、按象限分支等额外自由度，避免 2-4 个样例无法归纳。
#     - 同时保持 2-3 个 mover，让解题者能看到“多个不同源颜色 → 同一框架基色”的重复支撑。
#   hidden_assumptions_audit:
#     - 原 solver 依赖 `col in move_colors` 从边框标记找到 mover；这个前提仍成立，因为输入标记规则不变。
#     - 原 solver 在印刻时执行 `out[...] = col`，默认“目标颜色 == 源 sprite 颜色”。
#       该前提在变体中失效，必须改为 `out[...] = frame_base`。
#     - 原 solver 不需要显式识别 frame_base；新变体需要，因此新增“从框架边界取出现次数最多的非背景色”。
#       生成器必须保证 frame_base 在边界上占严格多数。
#     - 原生成器只保证同一 mover 的多个目标不会互相重叠；变体中若不同 mover 在输出里都变同色，
#       相互接触会造成视觉混合，因此生成器额外保证不同 mover 的目标印刻 bbox 之间至少留 1 格间隔。
#   placement_legality:
#     - row/col marks 仍只选能让 sprite 完整落入 interior 的位置。
#     - 不同 mover 不共享行标记或列标记。
#     - 新增 non-touching bbox 检查，避免 recolor 后多个复制体连成一片。
#   visual_observability:
#     - 输入里 mover 颜色与框架基色都明显不同，输出里所有复制体统一变成框架基色，差异肉眼可见。
#     - frame_base 随样例变化，因此不会退化成“总是某个固定颜色”的不可解释常数规则。
#   cleanliness:
#     - 框架内部只出现统一颜色的 blob 复制体，不允许它们互相接触，减少视觉噪声。
#     - distractor 仍留在左侧，帮助区分“会被搬移的 sprite”和“不会参与的 sprite”。
#   color_inference:
#     - 颜色变化的线索是：输出复制体颜色始终等于包围它们的框架主边框颜色，而不是等于标记颜色或原 sprite 颜色。
#     - 该规则在每个样例中由多个 mover 重复支持，足够从少量 I/O 对中推断。
#   solver_assumptions_after_change:
#     - 最大内部背景连通块仍唯一对应框架 interior。
#     - divider 仍是左侧满高单色列。
#     - frame_base 必须是框架边界最常见的非背景色；generator 按此构造。
#     - 每个 mover 颜色在边框上同时拥有行标记和列标记；否则它会被当成 distractor。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of largest connected component in `mask` that does not touch the grid border.
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
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _frame_base_color(g: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    """Most frequent non-bg color on the rectangular frame border."""
    border_vals: list[int] = []
    border_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    assert counts
    return int(counts.most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - Matching edge markers still choose target rows/cols by mover color.
    - The left-side mover is erased.
    - When stamped into the frame, the sprite keeps its shape but every stamped pixel is recolored
      to the frame's base border color instead of the mover's original color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

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
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

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
                    out[rr, cc3] = int(frame_base)

    return out


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
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _stamp_bbox(sp: np.ndarray, tr: int, tc: int) -> tuple[int, int, int, int]:
    sh, sw = sp.shape
    r0 = int(tr - sh // 2)
    c0 = int(tc - sw // 2)
    return int(r0), int(c0), int(r0 + sh - 1), int(c0 + sw - 1)


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _pick_targets(
    sp: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    occupied_rows: set[int],
    occupied_cols: set[int],
    existing_bboxes: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]] | None:
    """Pick row/col marks that fit the sprite and keep all stamped copies visually separate."""
    sh, sw = sp.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None

    avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
    avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
    if not avail_rows or not avail_cols:
        return None

    for _ in range(80):
        want_rows = random.randint(1, 2)
        want_cols = random.randint(1, 2)
        row_marks = _pick_spaced(avail_rows[:], want_rows, sh)
        col_marks = _pick_spaced(avail_cols[:], want_cols, sw)
        if not row_marks or not col_marks:
            continue

        new_boxes = [_stamp_bbox(sp, r, c) for r in row_marks for c in col_marks]
        ok = True
        for i in range(len(new_boxes)):
            for j in range(i + 1, len(new_boxes)):
                if _bboxes_too_close(new_boxes[i], new_boxes[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for box in new_boxes:
            if any(_bboxes_too_close(box, prev) for prev in existing_bboxes):
                ok = False
                break
        if ok:
            return sorted(row_marks), sorted(col_marks), new_boxes
    return None


def _variant_metric(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """
    Return:
    - number of recolored stamped pixels inside the frame interior
    - number of mover colors whose output color changed away from source color
    """
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

    divider_mask = np.zeros_like(g, dtype=bool)
    for c in range(fc0):
        colv = g[:, c]
        non = colv[colv != bg]
        if len(non) >= g.shape[0] - 1 and len(set(map(int, non))) == 1:
            divider_mask[:, c] = True
    left_cols = np.arange(g.shape[1])[None, :] < fc0

    left_colors: set[int] = set()
    for col in sorted(set(map(int, g.flatten())) - {bg}):
        mask = (g == col) & left_cols & (~divider_mask)
        if not mask.any():
            continue
        left_colors.add(int(col))

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(ir0, ir1 + 1):
        cl = int(g[r, fc0])
        cr = int(g[r, fc1])
        if cl == cr and cl != bg and cl in left_colors:
            row_marks[cl].append(r)
    for c in range(ic0, ic1 + 1):
        ct = int(g[fr0, c])
        cb = int(g[fr1, c])
        if ct == cb and ct != bg and ct in left_colors:
            col_marks[ct].append(c)
    mover_colors = set(row_marks.keys()) & set(col_marks.keys())

    changed_colors = 0
    recolored_pixels = 0
    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = True
    for col in mover_colors:
        if np.any(out[interior_mask] == col):
            continue
        changed_colors += 1
    recolored_pixels = int(np.sum((out == frame_base) & interior_mask & (g != frame_base)))
    return recolored_pixels, changed_colors


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

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    occupied_target_boxes: list[tuple[int, int, int, int]] = []
    for mc, sp in zip(mover_colors, sprites):
        picked = _pick_targets(sp, ir0, ic0, ir1, ic1, occupied_rows, occupied_cols, occupied_target_boxes)
        if picked is None:
            return None
        row_marks, col_marks, boxes = picked
        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)
        occupied_target_boxes.extend(boxes)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if r in rmarks:
                val = int(mc)
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for mc, sp, _is_distractor in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None

    recolored_pixels, changed_colors = _variant_metric(inp, output)
    if recolored_pixels <= 0:
        return None
    if changed_colors < n_movers:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
