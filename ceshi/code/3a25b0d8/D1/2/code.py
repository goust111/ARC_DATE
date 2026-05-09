# Variant of 3a25b0d8
# Dimension: D1 — 匹配条件：排列匹配方式 (match_cond)
# Change: arrangement_match — src 子精灵与 dst 洞按质心的行列秩序(row_rank, col_rank)匹配
#         -> 面积/大小排序匹配：src 子精灵按面积降序与 dst 洞按面积降序一一对应

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location

"""
design_analysis:
  a_concretization_and_trace:
    input_shape:
      - 画面中仍然只有两个 8 连通对象：一个 src（轮廓内填了彩色子精灵），一个 dst（同色轮廓内是空洞）。
      - 输出仍然是把 dst 的洞填满后，对 dst 做 tight crop。
      - 变体的关键不再是“左上那个颜色去填左上那个洞”，而是“最大色块填最大洞，第二大填第二大，依次类推”。
    small_trace:
      - 假设 src 从左到右三个子精灵面积分别为 4/9/6，对应颜色为 红/蓝/绿。
      - 假设 dst 从左到右三个洞面积分别为 7/3/10。
      - 原题按位置配：红->左洞，蓝->中洞，绿->右洞。
      - 本变体按面积配：src 面积降序是 蓝(9), 绿(6), 红(4)；dst 面积降序是 右洞(10), 左洞(7), 中洞(3)。
      - 所以输出应为：右洞填蓝，左洞填绿，中洞填红。
      - 差异点发生在“匹配/配对”环节，洞提取、对象定位、tight crop 都不变。
  b_complexity_calibration:
    - 如果继续沿用原题的严格左右镜像布局，左右洞常常同面积，会让面积排序大量打平，规则退化不可观察。
    - 因此生成器保留“紧凑多腔体 + 共用轮廓色 + 两对象随机放置”的整体框架，但取消强制镜像同面积约束。
    - 为了让人眼可推断，洞数控制在 4-7，每个洞/子精灵都使用不同面积，且颜色也唯一，避免排序和颜色同时打平。
  c_hidden_assumptions_in_original_main:
    - 原 `_rank_keys()` 和 `_transform_point()` 依赖旧前提：src 与 dst 的对应关系能由质心行列秩序恢复。
    - 这个前提在新变体中失效，因为同一位置未必对应同一颜色，位置只用于展示，不参与匹配。
    - 原代码寻找旋转/翻转 `best=(k,flip)` 也是基于位置模式匹配；新规则只看面积，因此这一段必须删除，否则会引入无关假设。
    - 原 `_colored_components()` 只记录颜色和质心；新规则需要面积，因此这里必须补 `size`。
    - 原最终 `zip(src_list, dest_list)` 默认两个列表按位置排序；现在必须改成按面积排序。
  d_placement_legality:
    - 仍使用 `random_free_location()` 在 22-30 画布上放置两个对象，避免越界和像素重叠。
    - 对象内部使用 3x3 槽位紧凑排布，行列间隔 1 格，使 outline 容易连成单一 8 连通对象，同时不同洞在 4 连通下保持分离。
  e_visual_observability:
    - 面积排序必须能在 2-3 个样例里直接看出来，因此禁止用颜色编号、生成顺序等不可见信息参与匹配。
    - 生成器强制让 src 的“面积顺序”与 dst 的“面积顺序”对应，但二者的空间顺序通常不同，从而让原题规则与新规则产生可见分叉。
  f_visual_cleanliness:
    - 每个洞都在独立 3x3 槽位内，轮廓与彩色块层次清楚，没有额外标记叠加。
    - 行列间距固定为 1，既足够紧凑形成单对象，也不会挤到难以分辨。
  g_color_inference:
    - 每个子精灵颜色唯一，颜色只承担“身份标签”作用；真正决定填充目标的是面积排序。
    - 颜色数限制为 4-7，与洞数一致，避免重复颜色掩盖规则。
  h_generator_non_degeneration_checks:
    - 显式拒绝 `src_rank_per_slot == dst_rank_per_slot` 的样本，因为那会退化成“按位置填也恰好正确”。
    - 同时要求 src/dst 的面积排列都不是按槽位天然升降序，保证尺寸比较是必要步骤。
  i_new_main_assumptions_and_generator_guarantees:
    - 新 `main()` 假设 src 子精灵数量与 dst 洞数量一致，且面积几乎不会打平。
    - 生成器通过“4-7 个互不相同的面积值”保证这一点。
    - 新 `main()` 的 tie-break 只作为健壮性后备；真正的任务语义由唯一面积决定。
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill destination holes using source colors matched by component area.

    The two objects are still detected exactly as in the original task:
    - src = filled cavities
    - dst = empty cavities

    The only semantic change is the matching rule:
    - sort src colored subsprites by area descending
    - sort dst holes by area descending
    - fill largest hole with largest subsprite's color, etc.
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
        col = int(src_comp["color"])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest)


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 7)
ROW_GAP = 1
COL_GAP = 1
SLOT_H = 3
SLOT_W = 3
AREA_CHOICES = list(range(3, 10))  # 3..9, enough for 7 unique areas in a 3x3 slot

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 400


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


def _slot_columns(k: int) -> list[int]:
    if int(k) == 1:
        return [4]
    if int(k) == 2:
        return random.choice([[0, 4], [4, 8], [0, 8]])
    return [0, 4, 8]


def _row_partitions(total: int, rows: int) -> list[list[int]]:
    out: list[list[int]] = []

    def rec(rem: int, left: int, cur: list[int]) -> None:
        if left == 0:
            if rem == 0:
                out.append(cur.copy())
            return
        for k in (1, 2, 3):
            if k > rem:
                continue
            if rem - k > 3 * (left - 1):
                continue
            if rem - k < 1 * (left - 1):
                continue
            cur.append(k)
            rec(rem - k, left - 1, cur)
            cur.pop()

    rec(int(total), int(rows), [])
    return out


def _design_layout(n_target: int) -> list[dict]:
    """Compact multi-row slot layout. Slots are row-major and share 3x3 bboxes."""
    candidates = _row_partitions(int(n_target), 2) + _row_partitions(int(n_target), 3)
    rows = random.choice(candidates)
    layout: list[dict] = []
    cur_r = 0
    for k in rows:
        for col in _slot_columns(int(k)):
            layout.append({"r0": int(cur_r), "col_offset": int(col)})
        cur_r += SLOT_H + ROW_GAP
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
    """Render an object patch and its always-filled counterpart."""
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        canvas[rr, cc] = int(colors[slot_idx])
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = int(outline_col)

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = int(bg)

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


def _sample_unique_areas(n: int) -> list[int]:
    vals = random.sample(AREA_CHOICES, int(n))
    vals.sort(reverse=True)
    return [int(v) for v in vals]


def _scrambled_rank_assignment(n: int) -> list[int]:
    base = list(range(int(n)))
    for _ in range(80):
        perm = base.copy()
        random.shuffle(perm)
        mismatch = sum(int(i != p) for i, p in enumerate(perm))
        if perm != base and mismatch >= max(2, int(n) // 2):
            return [int(v) for v in perm]
    random.shuffle(base)
    return [int(v) for v in base]


def _position_match_colors(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    src_colors: list[int],
) -> list[int]:
    """Simulate the original centroid-rank matching on the current layout/shapes."""
    src_items = []
    dst_items = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, src_shapes)):
        pts = np.argwhere(sh)
        src_items.append(
            (
                slot_idx,
                (
                    float(slot["r0"] + pts[:, 0].mean()),
                    float(slot["col_offset"] + pts[:, 1].mean()),
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
                    float(slot["r0"] + pts[:, 0].mean()),
                    float(slot["col_offset"] + pts[:, 1].mean()),
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
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    try:
        r, c = random_free_location(
            canvas,
            src,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, src, int(r), int(c), transparent=bg)
    try:
        r, c = random_free_location(
            canvas,
            dst,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, dst, int(r), int(c), transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    n_cavities = random.randint(*CAVITY_COUNT_RANGE)
    layout = _design_layout(n_cavities)

    # Unique colors keep the size-ranking correspondence human-readable.
    fill_pool = [int(c) for c in palette[1 : 1 + n_cavities]]

    src_rank_per_slot = _scrambled_rank_assignment(n_cavities)
    dst_rank_per_slot = _scrambled_rank_assignment(n_cavities)
    if src_rank_per_slot == dst_rank_per_slot:
        return None

    src_area_by_rank = _sample_unique_areas(n_cavities)
    dst_area_by_rank = _sample_unique_areas(n_cavities)

    src_slot_areas = [int(src_area_by_rank[r]) for r in src_rank_per_slot]
    dst_slot_areas = [int(dst_area_by_rank[r]) for r in dst_rank_per_slot]

    src_colors = [int(fill_pool[r]) for r in src_rank_per_slot]
    dst_expected_colors = [int(fill_pool[r]) for r in dst_rank_per_slot]

    # Reject positional-degenerate cases: under the original task, these would look correct too.
    if src_colors == dst_expected_colors:
        return None

    src_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, area) for area in src_slot_areas]
    dst_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, area) for area in dst_slot_areas]

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
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
