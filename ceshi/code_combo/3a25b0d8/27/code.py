# Combo variant of 3a25b0d8 (index 27)
# combo_spec: D1/2+D7/1+D9/1
# Name: 面积排序+大量空腔+全同色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste

"""
design_analysis:
  a_concretization_and_trace:
    input_shape:
      - 画面里仍然只有两个 8 连通对象：src 是有轮廓且腔体已被填色的对象，dst 是同轮廓色但腔体为空的对象。
      - 本融合版把腔体数扩到 10-15 个，并把所有 src 子精灵统一成同一种填充色。
      - 求解时仍然先识别 src/dst，再抽取 src 的彩色 4 连通子精灵和 dst 的 bg 洞。
    small_trace:
      - 设 src 有 5 个同色子精灵，面积依次为 9/8/7/6/5；dst 有 5 个洞，面积依次为 8/5/9/6/7。
      - 原题会按质心行列秩序去配对，D1/2 则改为面积降序配对：9->9, 8->8, 7->7, 6->6, 5->5。
      - 但 D9/1 要求所有 src 子精灵同色，因此无论配对次序怎样，最终 dst 每个洞都会被填成同一个颜色。
      - 也就是说：输入结构上确实实例化了“面积排序配对”，但输出像素层面被“全同色”折叠成统一着色。
  b_complexity_calibration:
    - 为了落实 D7/1，生成器固定产生 10-15 个腔体，并用 4-5 行紧凑槽位承载。
    - 为了让“面积排序”在输入侧仍然真实存在，src/dst 使用同一批近似面积值的不同排列；大量面积只在 2..9 之间波动，制造“面积接近、数量很多”的观察负担。
    - 为了避免对象大到无法摆放，槽位维持 3x3，小面积重复允许存在；这和 D9/1 的单色输出是相容的，因为 tie 不会影响可见答案。
  c_hidden_assumptions_in_original_main:
    - 原 `_rank_keys()` / `_transform_point()` 假设对应关系来自几何秩序；D1/2 后这个前提失效，因此整段位置匹配逻辑必须删除。
    - 原 `_colored_components()` 只需要颜色和质心；现在必须记录面积。
    - 原 `_outline_col()` 用“最多的非 bg 颜色”猜轮廓色；D9/1 下所有子精灵同色且数量很多，填充色总面积可能超过轮廓色，这个假设失效。
    - 新实现改为：优先用对象 tight bbox 边界上的非 bg 颜色识别轮廓色。因为轮廓一定接触 bbox 边界，而腔体填色不会接触边界。
    - 原最终 `zip(src_list, dest_list)` 依赖按位置排序后的列表；现在必须改成按面积排序，tie 时只用质心做稳定性后备。
  d_placement_legality:
    - 仍然只生成两个对象，并保证非 bg 像素不重叠、不越界。
    - 由于 10-15 个腔体会显著提高轮廓面积，如果画布过小就会破坏“bg 是全局众数”这一原题前提。
    - 因此画布放宽到 34-38，并用左右分栏放置，稳定保住背景众数和双对象分离。
  e_visual_observability:
    - 这个组合存在先天冲突：D9/1 让所有洞最终同色，输出无法直接暴露 D1/2 的配对结果。
    - 因此这里把“可观察性”落在输入结构上：样例会稳定呈现大量同色腔体、src/dst 面积排列明显打散、且面积值集中在狭窄区间。
    - 人类从样例最容易读出的表层规则会变成“把单色 src 的颜色灌进 dst 全部洞里”；面积排序则是被要求保留的潜在内部机制。
  f_visual_cleanliness:
    - 所有腔体都放在规则槽位里，轮廓单层包边，没有额外标记，不会出现多种编码叠在同一局部区域的情况。
    - 行列间距固定 1，使对象仍是单一 8 连通体，同时各腔体在 4 连通下分开。
  g_change_inferability:
    - 颜色不再承担身份标记，只剩“统一填充材料”的作用。
    - 为了不让图面完全失去结构，生成器强制 src/dst 的面积排列有较高错位率，并尽量使用相近面积，突出“很多腔体、需要数面积”的题面观感。
  h_generator_non_degeneration_checks:
    - 由于 D9/1，`main()` 的可见输出对“面积配对”和“位置配对”会收敛为同一个单色结果；这是组合本身带来的语义折叠，不是生成 bug。
    - 因此生成器用结构指标而不是输出颜色多样性做约束：
      - 腔体数必须在 10-15。
      - src/dst 槽位面积赋值的海明错位数至少达到 `max(6, n//2)`。
      - 所有 src 腔体颜色完全一致。
      - 面积值至少覆盖 4 个不同档位，且都落在 2..9 的狭窄区间内。
  i_new_main_assumptions_and_generator_guarantees:
    - 新 `main()` 假设存在两个主要 8 连通对象，src 的彩色子精灵数与 dst 的洞数一致。
    - 生成器通过构造和 oracle 检查保证上述条件。
    - `main()` 在面积 tie 时按质心排序仅为稳定输出；生成器不要求面积全唯一。
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill destination holes using source colors matched by component area.

    Relative to the original task, the only semantic change is the pairing rule:
    source colored components and destination holes are both sorted by area
    (largest first) before colors are transferred.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _components(mask: np.ndarray) -> list[dict]:
        lab, n = label_mask(mask, connectivity=8)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "cid": int(cid),
                    "pts": pts,
                    "size": int(len(pts)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _extract_two_objects() -> list[list[dict]]:
        comps = _components(g != int(bg))
        if len(comps) < 2:
            return [comps]
        return [[comps[0]], [comps[1]]]

    def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
        r0 = min(int(d["bbox"][0]) for d in group)
        c0 = min(int(d["bbox"][1]) for d in group)
        r1 = max(int(d["bbox"][2]) for d in group)
        c1 = max(int(d["bbox"][3]) for d in group)
        return int(r0), int(c0), int(r1), int(c1)

    def _extract_group_subgrid(group: list[dict]) -> np.ndarray:
        r0, c0, r1, c1 = _bbox_of(group)
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for d in group:
            for rr, cc in d["pts"]:
                sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _tight_crop_non_bg(sub: np.ndarray) -> np.ndarray:
        mask = sub != int(bg)
        pts = np.argwhere(mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        mask = sub == int(bg)
        lab, n = label_mask(mask, connectivity=4)
        hs, ws = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(hs - 1) or c1 == int(ws - 1)):
                continue
            holes.append(
                {
                    "pts": pts,
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        border_vals = []
        if sub.size:
            border_vals.extend(int(v) for v in sub[0, :])
            border_vals.extend(int(v) for v in sub[-1, :])
            border_vals.extend(int(v) for v in sub[:, 0])
            border_vals.extend(int(v) for v in sub[:, -1])
        border_non_bg = [int(v) for v in border_vals if int(v) != int(bg)]
        if border_non_bg:
            return int(Counter(border_non_bg).most_common(1)[0][0])
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        mask = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(mask, connectivity=4)
        subs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            subs.append(
                {
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "color": int(col),
                }
            )
        return subs

    groups = _extract_two_objects()
    assert not (len(groups) < 2)
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(obj) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest)

    dest_list = sorted(
        dest_holes,
        key=lambda h: (-int(h["size"]), float(h["cent"][0]), float(h["cent"][1])),
    )
    src_list = sorted(
        src_subs,
        key=lambda s: (-int(s["size"]), float(s["cent"][0]), float(s["cent"][1]), int(s["color"])),
    )

    for src_comp, hole in zip(src_list, dest_list):
        col = int(src_comp["color"])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest)


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (34, 38)
OBJECT_H_RANGE = (13, 21)
OBJECT_W_RANGE = (9, 13)

CAVITY_COUNT_RANGE = (10, 15)
ROW_GAP = 1
SLOT_H = 3
SLOT_W = 3
LAYOUT_ROWS = 5
LAYOUT_COLS = 3
AREA_CHOICES = list(range(2, 10))

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
MAX_GEN_ATTEMPTS = 800


def _grow_connected_shape(h: int, w: int, area: int) -> np.ndarray:
    """Random connected polyomino of exact area inside an h x w slot."""
    if not (1 <= int(area) <= int(h * w)):
        raise ValueError("area out of range")
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(sorted(frontier)))
    out = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        out[int(r), int(c)] = True
    return out


def _design_layout(n_target: int) -> list[dict]:
    """Choose n_target occupied slots from a 4x4 slot board while keeping slot adjacency 8-connected."""
    start = (random.randrange(LAYOUT_ROWS), random.randrange(LAYOUT_COLS))
    chosen = {start}
    frontier = {start}
    while len(chosen) < int(n_target):
        cell = random.choice(sorted(frontier))
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(cell[0] + dr), int(cell[1] + dc)
                if 0 <= nr < LAYOUT_ROWS and 0 <= nc < LAYOUT_COLS and (nr, nc) not in chosen:
                    nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(cell)
            if not frontier:
                frontier = chosen.copy()
            continue
        pick = random.choice(nbrs)
        chosen.add(pick)
        frontier.add(pick)
    layout: list[dict] = []
    for rr, cc in sorted(chosen):
        layout.append({"r0": int(rr * (SLOT_H + ROW_GAP)), "col_offset": int(cc * (SLOT_W + 1))})
    return layout


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Render a fixed chamber-frame object and its always-filled counterpart."""
    if not layout:
        return None, None

    h = LAYOUT_ROWS * (SLOT_H + ROW_GAP) + 1
    w = LAYOUT_COLS * (SLOT_W + 1) + 1
    canvas = np.full((h, w), int(outline_col), dtype=int)
    filled_grid = np.full((h, w), int(outline_col), dtype=int)

    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        r0 = 1 + int(slot["r0"])
        c0 = 1 + int(slot["col_offset"])
        chamber = sh.astype(bool)
        if bool(chamber.any()):
            filled_grid[r0 : r0 + SLOT_H, c0 : c0 + SLOT_W][chamber] = int(colors[slot_idx])
            if filled:
                canvas[r0 : r0 + SLOT_H, c0 : c0 + SLOT_W][chamber] = int(colors[slot_idx])
            else:
                canvas[r0 : r0 + SLOT_H, c0 : c0 + SLOT_W][chamber] = int(bg)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _sample_area_multiset(n: int) -> list[int]:
    """Sample many near-sized areas from a narrow band."""
    for _ in range(80):
        width = random.choice([4, 5, 6])
        start = random.randint(2, 10 - width)
        pool = list(range(start, start + width))
        areas = [int(random.choice(pool)) for _ in range(int(n))]
        if len(set(areas)) >= 4:
            return areas
    raise RuntimeError("failed to sample area multiset")


def _scrambled_assignment(values: list[int]) -> list[int]:
    base = list(values)
    for _ in range(80):
        perm = base.copy()
        random.shuffle(perm)
        mismatch = sum(int(a != b) for a, b in zip(base, perm))
        if mismatch >= max(6, len(base) // 2):
            return [int(v) for v in perm]
    random.shuffle(base)
    return [int(v) for v in base]


def _build_object_pair(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    src_colors: list[int],
    dst_expected_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)

    dst_cropped, dst_filled = _render_object(
        layout, dst_shapes, dst_expected_colors, bg, outline_col, filled=False
    )
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(
        layout, src_shapes, src_colors, bg, outline_col, filled=True
    )
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    pair_w = int(src.shape[1]) + int(dst.shape[1]) + PLACEMENT_PADDING + 2 * PLACEMENT_BORDER
    pair_h = max(int(src.shape[0]), int(dst.shape[0])) + 2 * PLACEMENT_BORDER
    if pair_w > canvas_w or pair_h > canvas_h:
        return None

    if random.random() < 0.5:
        left_patch, right_patch = src, dst
    else:
        left_patch, right_patch = dst, src

    left_c = PLACEMENT_BORDER
    right_c = canvas_w - PLACEMENT_BORDER - int(right_patch.shape[1])
    left_r = random.randint(PLACEMENT_BORDER, canvas_h - PLACEMENT_BORDER - int(left_patch.shape[0]))
    right_r = random.randint(PLACEMENT_BORDER, canvas_h - PLACEMENT_BORDER - int(right_patch.shape[0]))

    paste(canvas, left_patch, int(left_r), int(left_c), transparent=bg)
    paste(canvas, right_patch, int(right_r), int(right_c), transparent=bg)
    return canvas


def _structural_metrics(src_slot_areas: list[int], dst_slot_areas: list[int], src_colors: list[int]) -> dict[str, int]:
    return {
        "n_cavities": int(len(src_slot_areas)),
        "slot_area_mismatch": int(sum(int(a != b) for a, b in zip(src_slot_areas, dst_slot_areas))),
        "unique_area_levels": int(len(set(src_slot_areas + dst_slot_areas))),
        "unique_src_colors": int(len(set(src_colors))),
    }


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    fill_col = int(palette[1])

    n_cavities = random.randint(*CAVITY_COUNT_RANGE)
    layout = _design_layout(n_cavities)
    if len(layout) != n_cavities:
        return None

    base_areas = _sample_area_multiset(n_cavities)
    src_slot_areas = _scrambled_assignment(base_areas)
    dst_slot_areas = _scrambled_assignment(base_areas)
    metrics = _structural_metrics(src_slot_areas, dst_slot_areas, [fill_col] * n_cavities)
    if metrics["slot_area_mismatch"] < max(6, n_cavities // 2):
        return None
    if metrics["unique_area_levels"] < 4:
        return None
    if metrics["unique_src_colors"] != 1:
        return None

    src_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, int(area)) for area in src_slot_areas]
    dst_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, int(area)) for area in dst_slot_areas]
    src_colors = [int(fill_col)] * n_cavities
    dst_expected_colors = [int(fill_col)] * n_cavities

    built = _build_object_pair(
        layout,
        src_shapes,
        dst_shapes,
        src_colors,
        dst_expected_colors,
        bg,
        outline_col,
    )
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
