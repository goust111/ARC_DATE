# Variant of 247ef758
# Dimension: D7 — 多步序列：覆盖顺序 (multi_step)
# Change: 按精灵在左侧条带中的位置从下到上排序（底部先印刻、顶部后印刻覆盖之前的）
# ... → 按精灵面积排序（面积小的先印刻，面积大的最后覆盖）
#
# Design analysis
# ---
# variant_spec:
#   unchanged_structure:
#     - 版式仍是 "左侧精灵条带 | 单列分隔条 | 右侧矩形框"
#     - 框四边标记的读取方式不变：左右边同色给出目标行，上下边同色给出目标列
#     - 每个 mover 仍会被擦除并按其颜色对应的 row_marks × col_marks 笛卡尔积居中印刻到框内
#   D7_core_change:
#     - 原题的覆盖顺序来自左侧条带里的垂直位置：更靠下者先印刻，更靠上者后印刻并覆盖
#     - 本变体改成面积序：像素数少的 mover 先印刻，像素数多的 mover 后印刻并覆盖
#     - 为避免面积并列导致学习信号变弱，generator 强制所有 mover 面积互异；solver 对外部输入仍提供稳定并列打破规则
#   concrete_pixel_behavior:
#     - 输入中仍能看到多个不同颜色的单色 blob 精灵
#     - 输出中如果两个 mover 的印刻区域发生重叠，最后可见的颜色不再由“谁在左条带更靠上”决定，而由“谁面积更大”决定
#     - 例：上方是 3 像素小精灵、下方是 8 像素大精灵，且两者在框内目标中心只差 (1,1)
#       原题：下方大精灵先印，小精灵后印，小精灵覆盖局部重叠区
#       变体：小精灵先印，大精灵后印，大精灵覆盖同一重叠区
#   complexity_calibration:
#     - 仍保留 2-3 个 mover、0-1 个 distractor、1-2 个边缘标记的范围，不改题目的主体识别机制
#     - 但 generator 会把 mover 的“主目标中心”聚在同一片区域，提升重叠概率，让 D7 变化在少量样例中可见
#     - 若一个候选输入在“原顺序解”和“面积序解”下输出完全相同，则直接丢弃，避免退化成原题
#   implicit_assumption_review:
#     - 原 solver 中唯一直接依赖旧 D7 属性的代码是 movers.sort(key=lambda ...)
#     - 其他代码只负责：识别框内区域、识别 divider、抽取左条带精灵、由边缘标记求目标点、擦除原图、平移印刻
#     - 这些步骤对“按位置排序”没有强依赖，因此可以保留
#     - 新 solver 增加了一个隐式假设：若面积并列，学习者无法仅靠面积区分顺序；因此 generator 保证 mover 面积全不同
#   placement_legality:
#     - generator 先为每个 mover 计算“可被居中放到框内”的目标中心范围
#     - 再给每个 mover 选一组互不共享的主行/主列标记，并可选追加一个额外行/列标记
#     - 左条带中的精灵自上而下堆放且不重叠，所有印刻都裁剪在框内
#   visual_observability:
#     - 颜色值本身不参与排序，排序键只依赖可见的面积
#     - 左条带会刻意安排成“小在上，大在下”，使原题位置序与本题面积序相反
#     - 同时要求面积序输出 != 原位置序输出，从而保证样例里确实发生了可见的覆盖差异
#   visual_cleanliness:
#     - 只让 mover 在框内产生有限层数的重叠；distractor 仍留在左侧，避免把整个框填满
#     - 额外边缘标记仍然是镜像的简单单色点，不引入新符号系统

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Bbox of the largest connected component in `mask` that does not touch the border.
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


def _extract_scene(grid: np.ndarray) -> dict:
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
                    "mask": m,
                    "area": int(m.sum()),
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
        "grid": g,
        "bg": bg,
        "inner_bbox": (ir0, ic0, ir1, ic1),
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "sprites": sprites,
        "targets": targets,
    }


def _solve_with_order(grid: np.ndarray, *, order_mode: str) -> np.ndarray:
    """
    Shared solver core so the generator can compare the new area-order variant
    against the original position-order behavior and reject degenerate samples.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    sprites = scene["sprites"]
    targets = scene["targets"]

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if order_mode == "position":
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]), int(s["color"])))
    elif order_mode == "area_asc":
        # Primary key is visible area. Ties are resolved deterministically, but the
        # generator avoids tied areas so the learned rule remains "sort by area".
        movers.sort(
            key=lambda s: (
                int(s["area"]),
                ((int(s["r0"]) + int(s["r1"])) / 2.0),
                int(s["c0"]),
                int(s["color"]),
            )
        )
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
    Move colored sprites from the left strip into the framed area.

    Variant rule:
    - row/col targets are still read from matching opposite-edge markers
    - movers are still erased from the left and stamped centered at every target
    - overlap order is now by sprite area: smaller first, larger last
    """
    return _solve_with_order(grid, order_mode="area_asc")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    shuffled = list(candidates)
    random.shuffle(shuffled)
    picked: list[int] = []
    for v in shuffled:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int, n_pixels: int) -> np.ndarray | None:
    """
    Generate a cropped 8-connected single-color sprite with exactly `n_pixels`.
    """
    if max_h < 1 or max_w < 1 or n_pixels < 1 or n_pixels > max_h * max_w:
        return None

    min_h = 2 if max_h >= 2 else 1
    min_w = 2 if max_w >= 2 else 1
    for _ in range(80):
        sh = random.randint(min_h, max_h)
        sw = random.randint(min_w, max_w)
        if sh * sw < n_pixels:
            continue

        pixels = {(sh // 2, sw // 2)}
        while len(pixels) < n_pixels:
            frontier = set()
            for r, c in pixels:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < sh and 0 <= nc < sw and (nr, nc) not in pixels:
                            frontier.add((nr, nc))
            if not frontier:
                break
            pixels.add(random.choice(tuple(frontier)))

        if len(pixels) != n_pixels:
            continue

        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        for pr, pc in pixels:
            sprite[pr - min_r, pc - min_c] = color
        return sprite

    return None


def _center_bounds(ir0: int, ic0: int, ir1: int, ic1: int, sprite: np.ndarray) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    return min_row, max_row, min_col, max_col


def _choose_cluster_centers(
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    sprites: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    n = len(sprites)
    row_offsets = list(range(n))
    col_offsets = list(range(n))

    row_anchor_lo = -10**9
    row_anchor_hi = 10**9
    col_anchor_lo = -10**9
    col_anchor_hi = 10**9
    for sp, ro, co in zip(sprites, row_offsets, col_offsets):
        min_row, max_row, min_col, max_col = _center_bounds(ir0, ic0, ir1, ic1, sp)
        row_anchor_lo = max(row_anchor_lo, int(min_row - ro))
        row_anchor_hi = min(row_anchor_hi, int(max_row - ro))
        col_anchor_lo = max(col_anchor_lo, int(min_col - co))
        col_anchor_hi = min(col_anchor_hi, int(max_col - co))
    if row_anchor_lo > row_anchor_hi or col_anchor_lo > col_anchor_hi:
        return None

    anchor_r = random.randint(int(row_anchor_lo), int(row_anchor_hi))
    anchor_c = random.randint(int(col_anchor_lo), int(col_anchor_hi))
    return [(int(anchor_r + ro), int(anchor_c + co)) for ro, co in zip(row_offsets, col_offsets)]


def _add_optional_extra_marks(
    primary: int,
    candidates: list[int],
    occupied: set[int],
    min_spacing: int,
) -> list[int]:
    marks = [int(primary)]
    if random.random() < 0.45:
        extra_candidates = [v for v in candidates if v not in occupied and abs(v - primary) >= min_spacing]
        if extra_candidates:
            extra = random.choice(extra_candidates)
            marks.append(int(extra))
            occupied.add(int(extra))
    return sorted(marks)


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

    # Distinct areas keep the D7 signal legible and make area-order unique.
    area_choices = list(range(3, 10))
    if len(area_choices) < n_movers:
        return None
    random.shuffle(area_choices)
    mover_areas = sorted(area_choices[:n_movers])

    mover_info: list[dict] = []
    for mc, area in zip(mover_colors, mover_areas):
        sp = _make_sprite(mc, 5, sprite_area_w, area)
        if sp is None:
            return None
        mover_info.append({"color": int(mc), "sprite": sp, "area": int(area)})

    # Small-on-top / large-on-bottom ensures the original position order conflicts
    # with the new area order whenever movers overlap in the frame.
    mover_info.sort(key=lambda item: (int(item["area"]), int(item["color"])))
    mover_sprites = [item["sprite"] for item in mover_info]
    centers = _choose_cluster_centers(ir0, ic0, ir1, ic1, mover_sprites)
    if centers is None:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for item, (primary_r, primary_c) in zip(mover_info, centers):
        mc = int(item["color"])
        sp = item["sprite"]
        sh, sw = sp.shape
        min_row, max_row, min_col, max_col = _center_bounds(ir0, ic0, ir1, ic1, sp)
        if not (min_row <= primary_r <= max_row and min_col <= primary_c <= max_col):
            return None
        if primary_r in occupied_rows or primary_c in occupied_cols:
            return None
        occupied_rows.add(int(primary_r))
        occupied_cols.add(int(primary_c))
        row_marks = _add_optional_extra_marks(
            int(primary_r),
            list(range(int(min_row), int(max_row) + 1)),
            occupied_rows,
            max(2, int(sh)),
        )
        col_marks = _add_optional_extra_marks(
            int(primary_c),
            list(range(int(min_col), int(max_col) + 1)),
            occupied_cols,
            max(2, int(sw)),
        )
        mover_targets[mc] = (row_marks, col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for item in mover_info:
            mc = int(item["color"])
            rmarks, _ = mover_targets[mc]
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = int(val)
        inp[r, fc1] = int(val)

    cur_row = 0
    for item in mover_info:
        sp = item["sprite"]
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += int(sh)

    for dc_color in distractor_colors:
        remaining_h = H - cur_row
        if remaining_h <= 0:
            return None
        d_area = random.randint(3, min(7, max(3, remaining_h * sprite_area_w)))
        ds = _make_sprite(int(dc_color), min(5, remaining_h), sprite_area_w, d_area)
        if ds is None:
            return None
        sh, sw = ds.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if ds[dr, dc] != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(ds[dr, dc])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        area_out = main(inp)
        position_out = _solve_with_order(inp, order_mode="position")
    except Exception:
        return None

    if np.array_equal(inp, area_out):
        return None
    if np.array_equal(area_out, position_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
