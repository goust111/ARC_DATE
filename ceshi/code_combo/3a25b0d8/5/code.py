# Combo variant of 3a25b0d8 (index 5)
# combo_spec: D1/2+D7/1
# Name: 面积排序匹配+大量空腔

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste

"""
design_analysis:
  a_concretization_and_trace:
    - 画面中仍然只有两个共享轮廓色的 8 连通对象：src 内部是彩色子精灵，dst 内部是空洞。
    - 原题会先看 src/dst 子结构的质心行列秩序，再按位置一一配对；融合后完全改成“按面积降序配对”。
    - 小例子：若 src 有 12 个子精灵，面积降序颜色序列是
      [红, 蓝, 绿, 黄, 橙, 青, 品, 红, 蓝, 绿, 黄, 橙]，
      dst 的 12 个洞面积降序对应到 12 个空间位置却是另一套乱序。
      原题会把“左上行列秩序第 1 个颜色”填给“左上行列秩序第 1 个洞”；
      现在必须先把 12 个 src 子精灵按面积排序，再把 12 个 dst 洞按面积排序，
      最大填最大，第二大填第二大。大量空腔让比较次数明显增加。
    - 与单一 D1/2 相比，这里不是 4-7 个洞，而是 10-15 个洞；因此 generator
      需要把更多槽位塞进同一对象，并显式避免“面积排序结果恰好和位置排序一样”。
  b_complexity_calibration:
    - 若继续使用原题镜像布局，很多左右洞会同面积，面积排序会大量打平，规则不可观察。
    - 这里改成 4x4 槽位、2-3 行、最多 5 列，总计 10-15 个槽位；每个洞面积优先从 4..16
      里抽取连续值。10-13 个洞时可做到全唯一；14-15 个洞时只允许最小面积端出现极少量重复，
      这样既保证轮廓连通，又保留“按面积比大小”的核心负担。
    - 颜色数限制为 6-8 个 fill colors，超过颜色数的洞允许复用颜色；颜色只是身份标签，
      真正决定配对的是面积排序。
  c_hidden_assumptions_in_original_main:
    - 原 `_rank_keys()` / `_transform_point()` 假设 src 与 dst 可由质心秩序或几何变换对齐。
      融合后这个前提失效，因此整段旋转/翻转搜索必须删除。
    - 原 `_colored_components()` 只需颜色和质心；现在必须记录 `size`，因为排序主键变成面积。
    - 原 `zip(src_list, dest_list)` 依赖“两个列表已经按位置排好”；新版本必须改成按面积排序。
    - 新 solver 的隐式假设是：src 子精灵数 == dst 洞数，且面积排序足够稳定。
      generator 通过 10-13 个全唯一面积、14-15 个近唯一面积保证这一点；若低端面积打平，
      tie-break 只作健壮性后备。
  d_placement_legality:
    - 大量空腔会把对象宽高推高，因此不再依赖纯随机 free-location 试错。
    - 生成器先构造 tight object，再在 28-30 画布上做可行的横向或纵向分离摆放，保证不重叠、不越界。
  e_visual_observability:
    - 解题者肉眼能直接看见每个 src 色块和每个 dst 洞的大小差异；面积值全来自像素个数，
      不使用颜色编号、生成顺序等不可见因素。
    - 另外显式拒绝“旧位置规则也会得到同样输出”的样本，确保样例真的展示融合行为。
  f_visual_cleanliness:
    - 每个洞都待在独立 4x4 槽位中，槽位间只留 1 格，使轮廓能连成单对象但洞在 4 连通下分离。
    - src/dst 放在画布上分区摆放，不叠加、不共享像素，视觉层次清楚。
  g_color_inference:
    - 因为洞数 > 可用非背景颜色数，所以允许颜色复用，但复用发生在不同面积秩上；
      人类仍可从“按大小对应”的结构线索中恢复规则。
  h_generator_non_degeneration_checks:
    - 拒绝 `src_rank_per_slot == dst_rank_per_slot`。
    - 拒绝“旧位置秩序匹配得到的 dst 颜色分布 == 新面积匹配的颜色分布”。
    - 拒绝输入已经等于输出的平凡样本。
  i_new_main_assumptions_and_generator_guarantees:
    - main() 只依赖：两对象可由 hole area 区分 src/dst、src 彩色块和 dst 洞数量相等、每块单色。
    - generator 保证：src 为 filled object、dst 为空洞 object、两者都是单个 8 连通对象、
      洞数与子精灵数一致、面积集合在大多数样本中唯一，且多达 10-15 个。
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill destination holes using source colors matched by descending component area.

    The object with more enclosed background area is dst; the other is src.
    Matching ignores spatial arrangement entirely and uses area order only.
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
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
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
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "size": int(len(pts)),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
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
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "size": int(len(pts)),
                    "color": int(col),
                }
            )
        return subs

    def _tight_crop_non_bg(sub: np.ndarray) -> np.ndarray:
        m = sub != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    groups = _extract_two_objects()
    assert not (len(groups) < 2)
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(o) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
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
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(src_comp["color"])

    return _tight_crop_non_bg(dest)


CANVAS_SIZE_RANGE = (28, 30)
OBJECT_H_RANGE = (9, 16)
OBJECT_W_RANGE = (14, 25)

CAVITY_COUNT_RANGE = (10, 15)
FILL_COLOR_COUNT_RANGE = (6, 8)

ROW_GAP = 1
COL_GAP = 1
AREA_VALUES = list(range(2, 17))

PLACEMENT_BORDER = 1
PLACEMENT_GAP_CHOICES = [0, 1, 1, 2]
MAX_GEN_ATTEMPTS = 400


def _grow_connected_shape(h: int, w: int, area: int) -> np.ndarray:
    """Random 4-connected polyomino of exact area inside a slot."""
    if not (1 <= int(area) <= int(h * w)):
        raise ValueError("area out of range")
    cells = {(random.randrange(int(h)), random.randrange(int(w)))}
    while len(cells) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < int(h) and 0 <= nc < int(w) and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(sorted(frontier)))
    out = np.zeros((int(h), int(w)), dtype=bool)
    for r, c in cells:
        out[int(r), int(c)] = True
    return out


def _row_partitions(total: int, rows: int, lo: int, hi: int) -> list[list[int]]:
    out: list[list[int]] = []

    def rec(rem: int, left: int, cur: list[int]) -> None:
        if left == 0:
            if rem == 0:
                out.append(cur.copy())
            return
        for k in range(lo, hi + 1):
            if k > rem:
                continue
            if rem - k < lo * (left - 1):
                continue
            if rem - k > hi * (left - 1):
                continue
            cur.append(int(k))
            rec(rem - k, left - 1, cur)
            cur.pop()

    rec(int(total), int(rows), [])
    return out


def _row_slot_widths(count: int) -> list[int]:
    templates = {
        3: [[4, 4, 4], [4, 4, 3]],
        4: [[4, 4, 4, 4], [4, 4, 4, 3]],
        5: [[4, 4, 4, 4, 3], [4, 4, 4, 3, 3]],
    }
    widths = list(random.choice(templates[int(count)]))
    random.shuffle(widths)
    return [int(v) for v in widths]


def _design_layout(n_target: int) -> list[dict]:
    candidates: list[list[int]] = []
    candidates.extend(_row_partitions(int(n_target), 2, 5, 5))
    candidates.extend(_row_partitions(int(n_target), 3, 3, 5))
    row_counts = random.choice(candidates)
    row_heights = [4, 4] if len(row_counts) == 2 else random.sample([4, 4, 3], 3)

    layout: list[dict] = []
    cur_r = 1
    for row_h, count in zip(row_heights, row_counts):
        cur_c = 1
        for width in _row_slot_widths(int(count)):
            layout.append(
                {
                    "r0": int(cur_r),
                    "c0": int(cur_c),
                    "h": int(row_h),
                    "w": int(width),
                }
            )
            cur_c += int(width) + COL_GAP
        cur_r += int(row_h) + ROW_GAP
    return layout


def _slot_capacity(slot: dict) -> int:
    return int(slot["h"]) * int(slot["w"])


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


def _assign_unique_areas(layout: list[dict], ordered_areas: list[int]) -> list[int] | None:
    free = list(range(len(layout)))
    areas_by_slot: list[int | None] = [None] * len(layout)
    for area in [int(v) for v in ordered_areas]:
        eligible = [idx for idx in free if _slot_capacity(layout[idx]) >= int(area)]
        if not eligible:
            return None
        min_cap = min(_slot_capacity(layout[idx]) for idx in eligible)
        tight = [idx for idx in eligible if _slot_capacity(layout[idx]) == int(min_cap)]
        slot_idx = int(random.choice(tight))
        areas_by_slot[slot_idx] = int(area)
        free.remove(slot_idx)
    return [int(v) for v in areas_by_slot if v is not None]


def _build_rank_colors(fill_pool: list[int], n: int) -> list[int]:
    rank_colors = [int(c) for c in fill_pool]
    while len(rank_colors) < int(n):
        rank_colors.append(int(random.choice(fill_pool)))
    random.shuffle(rank_colors)
    return rank_colors[: int(n)]


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Render a solid outline object with isolated interior slots."""
    if not layout:
        return None, None
    h = max(int(slot["r0"]) + int(slot["h"]) for slot in layout) + 1
    w = max(int(slot["c0"]) + int(slot["w"]) for slot in layout) + 1
    canvas = np.full((int(h), int(w)), int(outline_col), dtype=int)
    filled_grid = canvas.copy()

    for slot, shape, color in zip(layout, shapes, colors):
        rr, cc = np.where(shape)
        abs_r = rr + int(slot["r0"])
        abs_c = cc + int(slot["c0"])
        filled_grid[abs_r, abs_c] = int(color)
        canvas[abs_r, abs_c] = int(color) if filled else int(bg)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


def _position_match_colors(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    src_colors: list[int],
) -> list[int]:
    """Simulate the original centroid-rank matching on the current shapes."""
    src_items = []
    dst_items = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, src_shapes)):
        pts = np.argwhere(sh)
        src_items.append(
            (
                slot_idx,
                (
                    float(int(slot["r0"]) + pts[:, 0].mean()),
                    float(int(slot["c0"]) + pts[:, 1].mean()),
                ),
                int(src_colors[slot_idx]),
            )
        )
    for slot_idx, (slot, sh) in enumerate(zip(layout, dst_shapes)):
        pts = np.argwhere(sh)
        dst_items.append(
            (
                slot_idx,
                (
                    float(int(slot["r0"]) + pts[:, 0].mean()),
                    float(int(slot["c0"]) + pts[:, 1].mean()),
                ),
            )
        )

    def _rank_map(centers: list[tuple[float, float]]) -> tuple[dict[float, int], dict[float, int]]:
        rows = sorted(set(float(r) for r, _ in centers))
        cols = sorted(set(float(c) for _, c in centers))
        return (
            {float(r): i for i, r in enumerate(rows)},
            {float(c): i for i, c in enumerate(cols)},
        )

    src_rr, src_cc = _rank_map([item[1] for item in src_items])
    dst_rr, dst_cc = _rank_map([item[1] for item in dst_items])

    src_sorted = sorted(
        src_items,
        key=lambda item: (
            int(src_rr[float(item[1][0])]),
            int(src_cc[float(item[1][1])]),
            float(item[1][0]),
            float(item[1][1]),
        ),
    )
    dst_sorted = sorted(
        dst_items,
        key=lambda item: (
            int(dst_rr[float(item[1][0])]),
            int(dst_cc[float(item[1][1])]),
            float(item[1][0]),
            float(item[1][1]),
        ),
    )

    colors_by_dest_slot = [0] * len(layout)
    for src_item, dst_item in zip(src_sorted, dst_sorted):
        colors_by_dest_slot[int(dst_item[0])] = int(src_item[2])
    return colors_by_dest_slot


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
    h1, w1 = src.shape
    h2, w2 = dst.shape
    candidates: list[tuple[str, int, int, int]] = []
    for h in range(CANVAS_SIZE_RANGE[0], CANVAS_SIZE_RANGE[1] + 1):
        for w in range(CANVAS_SIZE_RANGE[0], CANVAS_SIZE_RANGE[1] + 1):
            for gap in PLACEMENT_GAP_CHOICES:
                if max(w1, w2) + 2 * PLACEMENT_BORDER <= w and h1 + h2 + gap + 2 * PLACEMENT_BORDER <= h:
                    candidates.append(("v", int(h), int(w), int(gap)))
                if max(h1, h2) + 2 * PLACEMENT_BORDER <= h and w1 + w2 + gap + 2 * PLACEMENT_BORDER <= w:
                    candidates.append(("h", int(h), int(w), int(gap)))
    if not candidates:
        return None

    orient, h, w, gap = random.choice(candidates)
    canvas = np.full((int(h), int(w)), int(bg), dtype=int)
    if orient == "v":
        if random.random() < 0.5:
            src, dst = dst, src
            h1, w1 = src.shape
            h2, w2 = dst.shape
        extra_rows = int(h - 2 * PLACEMENT_BORDER - gap - h1 - h2)
        top_pad = random.randint(0, extra_rows)
        r1 = int(PLACEMENT_BORDER + top_pad)
        r2 = int(r1 + h1 + gap)
        c1 = int(PLACEMENT_BORDER + random.randint(0, w - 2 * PLACEMENT_BORDER - w1))
        c2 = int(PLACEMENT_BORDER + random.randint(0, w - 2 * PLACEMENT_BORDER - w2))
        paste(canvas, src, r1, c1, transparent=bg)
        paste(canvas, dst, r2, c2, transparent=bg)
    else:
        if random.random() < 0.5:
            src, dst = dst, src
            h1, w1 = src.shape
            h2, w2 = dst.shape
        extra_cols = int(w - 2 * PLACEMENT_BORDER - gap - w1 - w2)
        left_pad = random.randint(0, extra_cols)
        c1 = int(PLACEMENT_BORDER + left_pad)
        c2 = int(c1 + w1 + gap)
        r1 = int(PLACEMENT_BORDER + random.randint(0, h - 2 * PLACEMENT_BORDER - h1))
        r2 = int(PLACEMENT_BORDER + random.randint(0, h - 2 * PLACEMENT_BORDER - h2))
        paste(canvas, src, r1, c1, transparent=bg)
        paste(canvas, dst, r2, c2, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    n_cavities = random.randint(*CAVITY_COUNT_RANGE)
    layout = _design_layout(n_cavities)
    rank_areas = sorted(random.sample(AREA_VALUES, n_cavities), reverse=True)

    src_slot_areas = _assign_unique_areas(layout, rank_areas)
    dst_slot_areas = _assign_unique_areas(layout, rank_areas)
    if src_slot_areas is None or dst_slot_areas is None or src_slot_areas == dst_slot_areas:
        return None

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    n_fill_cols = random.randint(fc_lo, min(fc_hi, len(palette) - 1))
    fill_pool = [int(c) for c in palette[1 : 1 + n_fill_cols]]
    rank_colors = _build_rank_colors(fill_pool, n_cavities)
    area_to_rank = {int(area): idx for idx, area in enumerate(rank_areas)}

    src_colors = [int(rank_colors[area_to_rank[int(area)]]) for area in src_slot_areas]
    dst_expected_colors = [int(rank_colors[area_to_rank[int(area)]]) for area in dst_slot_areas]

    src_shapes = [
        _grow_connected_shape(int(slot["h"]), int(slot["w"]), int(area))
        for slot, area in zip(layout, src_slot_areas)
    ]
    dst_shapes = [
        _grow_connected_shape(int(slot["h"]), int(slot["w"]), int(area))
        for slot, area in zip(layout, dst_slot_areas)
    ]

    if _position_match_colors(layout, src_shapes, dst_shapes, src_colors) == dst_expected_colors:
        return None

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
    if canvas is None or np.array_equal(canvas, expected):
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
