# Combo variant of 3a25b0d8 (index 3)
# combo_spec: D1/2+D9/1
# Name: 面积排序匹配+全同色空腔

from __future__ import annotations

from collections import Counter
import importlib.util
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location

"""
design_analysis:
  a_concretization_and_trace:
    - 输入仍然只有两个 8 连通对象：src 是“轮廓 + 彩色子精灵”，dst 是“轮廓 + 空洞”。
    - 原题匹配方式是按质心的行/列秩序配对；融合后改成按面积降序配对。
    - 同时所有 src 子精灵都改成同一颜色，所以颜色不再提供“哪块对应哪个洞”的线索。
    - 小例子：
      - src 四块面积为 9/7/5/3，颜色全是红。
      - dst 四个洞面积为 10/8/6/4。
      - 原题会按位置配；融合后必须按面积配：9->10, 7->8, 5->6, 3->4。
      - 因为颜色统一，最终看到的是“所有洞都被同一种红色填满”，但推理所依赖的信息已从位置切换到面积。
  b_complexity_calibration:
    - 如果继续沿用原始 LR 镜像布局，很多洞会天然同面积，面积排序就会打平，规则不可观察。
    - 为了同时满足“全同色”和“fill 像素要压过 outline 像素”这两个要求，生成器把腔体数收紧为 4 个，并改用 2x2 的 5x5 大槽位布局。
    - 每个样例里四个腔体面积仍然互不相同，且都从高面积区间采样，保证面积排序明显、同时提升 fill-majority 的可实现性。
    - 颜色只保留 1 个 fill color + 1 个 outline color，避免颜色自由度过高，同时保证“颜色不能帮助匹配”。
  c_hidden_assumptions_in_original_main:
    - 原 `_rank_keys()` / `_transform_point()` / 旋转翻转搜索依赖“位置模式决定配对”，D1/2 后全部失效，必须删除。
    - 原 `_outline_col()` 假设“最大非背景色就是 outline”。D9/1 后 fill 可能比 outline 更多，这个假设不再成立。
    - 原 `_colored_components()` 是通过“非背景且非 outline”找 src 子精灵；在全同色且 fill-majority 时会直接把真正的 fill 当成 outline 排除掉。
    - 新 solver 因此改成：
      1. 用 hole area 判断哪一个对象是 dst。
      2. 在 src 中按颜色逐色找“完全封闭、且不接触边界”的 4 连通组件。
      3. 优先选择“组件数 == dst 洞数”的那种颜色作为真正的 fill family。
      4. 对 src 与 dst 都按面积降序排序，再逐一填充。
  d_placement_legality:
    - 两对象仍通过 `random_free_location()` 独立放到 22-30 的画布上，带边界和 padding，不重叠、不越界。
    - 对象内部使用固定 2x2 槽位，四个腔体各占 5x5 的大槽位，槽位间保留 1 列/行距离；这样 cavity 在 4 连通下分离，同时整体仍可形成 8 连通对象。
  e_visual_observability:
    - 人眼能看到 src 里有若干同色小块，面积明显不同；dst 里有若干空洞，面积也明显不同。
    - 生成器显式拒绝“位置配对刚好等于面积配对”的样本，因此 D1/2 的控制变量在输入层面真实生效。
    - 由于所有 fill 同色，解题者必须依赖几何面积，而不是颜色标签。
  f_visual_cleanliness:
    - 整个画面只有背景 + 两个对象，没有额外标记。
    - 2x2 大槽位布局使每个腔体面积差更容易肉眼比较，也避免了过多小腔体挤在一起。
  g_change_inferrability:
    - 颜色变化规则极简：所有子精灵与所有被填洞都只有同一种 fill color。
    - 因而颜色本身没有任何映射信息；可推断结构只剩“这是封闭腔体”和“它的面积大小”。
  h_quantitative_anti_degeneration_check:
    - `behavior_report()` 统计两个量：
      - `position_vs_area_mismatch_rate`：同一输入上，旧的“按位置匹配”与新的“按面积匹配”在槽位级别有多常不同。
      - `seed_solver_diff_rate`：原始 solver 与本 combo solver 的输出不同的比例。
    - 前者验证 D1/2 真的生效；后者验证 D9/1 真的破坏了原 solver 的 outline 颜色假设，而不是仅仅做了外观改色。
  i_new_main_assumptions_and_generator_guarantees:
    - 新 main 假设：恰有两个 8 连通对象；dst 洞数与 src 内部 fill 组件数一致；src 真正的 fill 颜色在“封闭组件计数”上可识别；面积无平局。
    - 生成器保证：只生成两个对象；所有 src cavity 同色且 4 连通分离；dst 洞数等于 src cavity 数；腔体面积互异；fill area 压过 outline area，使原 solver 的 outline 颜色假设失效。
"""


def _load_seed_main():
    path = "/home/lixinyu/snap/new/arc2_solver_select/3a25b0d8.py"
    spec = importlib.util.spec_from_file_location("seed_3a25b0d8", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "main", None)


SEED_MAIN = _load_seed_main()


def _components(mask: np.ndarray, *, connectivity: int) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
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
    comps.sort(key=lambda comp: (-int(comp["size"]), float(comp["cent"][0]), float(comp["cent"][1])))
    return comps


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg), connectivity=8)
    assert len(comps) >= 2
    out = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        out.append(sub)
    return out


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(sub != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        holes.append(
            {
                "pts": pts,
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return holes


def _internal_color_components(sub: np.ndarray, bg: int) -> dict[int, list[dict]]:
    h, w = sub.shape
    result: dict[int, list[dict]] = {}
    for color in sorted(set(map(int, sub.flatten()))):
        if int(color) == int(bg):
            continue
        mask = sub == int(color)
        lab, n = label_mask(mask, connectivity=4)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
                continue
            comps.append(
                {
                    "pts": pts,
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": int(color),
                }
            )
        if comps:
            result[int(color)] = comps
    return result


def _color_components(sub: np.ndarray, color: int) -> list[dict]:
    mask = sub == int(color)
    lab, n = label_mask(mask, connectivity=4)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comps.append(
            {
                "pts": pts,
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": int(color),
            }
        )
    return comps


def _extract_src_dst(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_objects(g, bg)

    holes_per_obj = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_areas = [sum(int(h["size"]) for h in holes) for holes in holes_per_obj]
    dest_i = int(np.argmax(hole_areas))
    src_i = int(1 - dest_i)

    dest = objs[dest_i].copy()
    src = objs[src_i].copy()
    dest_holes = _enclosed_bg_components(dest, bg)

    best_color = None
    best_key = None
    need = len(dest_holes)
    for color in sorted(set(map(int, src.flatten()))):
        if int(color) == int(bg):
            continue
        comps = _color_components(src, int(color))
        trial = src.copy()
        trial[trial == int(color)] = int(bg)
        trial_holes = _enclosed_bg_components(trial, bg)
        internal = _internal_color_components(src, bg).get(int(color), [])
        key = (
            int(len(trial_holes) == need),
            int(len(comps) == need),
            int(len(internal) == need),
            int(len(trial_holes)),
            int(sum(int(comp["size"]) for comp in comps)),
            -int(color),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_color = int(color)
    src_subs = _color_components(src, int(best_color)) if best_color is not None else []
    return bg, src, dest, src_subs, dest_holes


def _sort_by_area(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (-int(item["size"]), float(item["cent"][0]), float(item["cent"][1])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solver for combo D1/2 + D9/1:
    - dst holes are identified as enclosed background 4-components
    - src subsprites are identified as enclosed monochrome 4-components
    - source and destination components are paired by descending area
    - all fills share the same color, but pairing logic is still area-based
    """
    bg, _src, dest, src_subs, dest_holes = _extract_src_dst(grid)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    src_list = _sort_by_area(src_subs)
    dest_list = _sort_by_area(dest_holes)
    for src_comp, hole in zip(src_list, dest_list):
        col = int(src_comp["color"])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)
    return _tight_crop_non_bg(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 4)
ROW_GAP = 1
SLOT_H = 5
SLOT_W = 5
AREA_CHOICES = list(range(14, 20))

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 500


def _grow_connected_shape(h: int, w: int, area: int) -> np.ndarray:
    if not (1 <= int(area) <= int(h * w)):
        raise ValueError("area out of range")
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r + dr), int(c + dc)
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
    assert int(n_target) == 4
    return [
        {"r0": 1, "col_offset": 1},
        {"r0": 1, "col_offset": 7},
        {"r0": 7, "col_offset": 1},
        {"r0": 7, "col_offset": 7},
    ]


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not layout:
        return None, None

    h = max(int(slot["r0"]) + SLOT_H for slot in layout) + 1
    w = max(int(slot["col_offset"]) + SLOT_W for slot in layout) + 1
    canvas = np.full((int(h), int(w)), int(outline_col), dtype=int)
    filled_grid = canvas.copy()

    for slot, shape, color in zip(layout, shapes, colors):
        rr, cc = np.where(shape)
        abs_r = rr + int(slot["r0"])
        abs_c = cc + int(slot["col_offset"])
        filled_grid[abs_r, abs_c] = int(color)
        canvas[abs_r, abs_c] = int(color) if filled else int(bg)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[int(r0b) : int(r1b) + 1, int(c0b) : int(c1b) + 1]
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


def _position_order(slot_items: list[tuple[int, tuple[float, float]]]) -> list[int]:
    rows = sorted(set(float(r) for _slot, (r, _c) in slot_items))
    cols = sorted(set(float(c) for _slot, (_r, c) in slot_items))
    rrank = {float(r): idx for idx, r in enumerate(rows)}
    crank = {float(c): idx for idx, c in enumerate(cols)}
    return [
        int(slot_idx)
        for slot_idx, _cent in sorted(
            slot_items,
            key=lambda item: (
                int(rrank[float(item[1][0])]),
                int(crank[float(item[1][1])]),
                float(item[1][0]),
                float(item[1][1]),
            ),
        )
    ]


def _position_match_ranks(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    src_rank_per_slot: list[int],
) -> list[int]:
    src_items = []
    dst_items = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, src_shapes)):
        pts = np.argwhere(sh)
        src_items.append(
            (
                int(slot_idx),
                (
                    float(slot["r0"] + pts[:, 0].mean()),
                    float(slot["col_offset"] + pts[:, 1].mean()),
                ),
            )
        )
    for slot_idx, (slot, sh) in enumerate(zip(layout, dst_shapes)):
        pts = np.argwhere(sh)
        dst_items.append(
            (
                int(slot_idx),
                (
                    float(slot["r0"] + pts[:, 0].mean()),
                    float(slot["col_offset"] + pts[:, 1].mean()),
                ),
            )
        )

    src_order = _position_order(src_items)
    dst_order = _position_order(dst_items)
    out = [0] * len(layout)
    for src_slot, dst_slot in zip(src_order, dst_order):
        out[int(dst_slot)] = int(src_rank_per_slot[int(src_slot)])
    return out


def _build_object_pair(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    fill_color: int,
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    colors = [int(fill_color)] * n_cavities

    dst_cropped, dst_filled = _render_object(
        layout, dst_shapes, colors, bg, outline_col, filled=False
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
        layout, src_shapes, colors, bg, outline_col, filled=True
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
    canvas_h = CANVAS_SIZE_RANGE[1]
    canvas_w = CANVAS_SIZE_RANGE[1]
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    placements = [
        ((PLACEMENT_BORDER, PLACEMENT_BORDER),
         (canvas_h - dst.shape[0] - PLACEMENT_BORDER, canvas_w - dst.shape[1] - PLACEMENT_BORDER)),
        ((PLACEMENT_BORDER, canvas_w - src.shape[1] - PLACEMENT_BORDER),
         (canvas_h - dst.shape[0] - PLACEMENT_BORDER, PLACEMENT_BORDER)),
    ]
    random.shuffle(placements)
    for (sr, sc), (dr, dc) in placements:
        if int(sr + src.shape[0] + PLACEMENT_PADDING) > int(dr):
            continue
        if max(int(sc), int(dc)) < min(int(sc + src.shape[1]), int(dc + dst.shape[1])) + PLACEMENT_PADDING:
            continue
        paste(canvas, src, int(sr), int(sc), transparent=bg)
        paste(canvas, dst, int(dr), int(dc), transparent=bg)
        return canvas
    return None


def _seed_differs(inp: np.ndarray, expected: np.ndarray) -> bool:
    if SEED_MAIN is None:
        return True
    try:
        baseline = SEED_MAIN(inp)
    except Exception:
        return True
    if not isinstance(baseline, np.ndarray):
        return True
    if baseline.shape != expected.shape:
        return True
    return not np.array_equal(baseline, expected)


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    fill_color = int(palette[1])

    n_cavities = random.randint(*CAVITY_COUNT_RANGE)
    layout = _design_layout(n_cavities)

    src_rank_per_slot = _scrambled_rank_assignment(n_cavities)
    dst_rank_per_slot = _scrambled_rank_assignment(n_cavities)
    if src_rank_per_slot == dst_rank_per_slot:
        return None

    src_area_by_rank = _sample_unique_areas(n_cavities)
    dst_area_by_rank = _sample_unique_areas(n_cavities)
    src_slot_areas = [int(src_area_by_rank[r]) for r in src_rank_per_slot]
    dst_slot_areas = [int(dst_area_by_rank[r]) for r in dst_rank_per_slot]

    src_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, area) for area in src_slot_areas]
    dst_shapes = [_grow_connected_shape(SLOT_H, SLOT_W, area) for area in dst_slot_areas]

    area_mapping = [int(r) for r in dst_rank_per_slot]
    pos_mapping = _position_match_ranks(layout, src_shapes, dst_shapes, src_rank_per_slot)
    if pos_mapping == area_mapping:
        return None

    built = _build_object_pair(
        layout,
        src_shapes,
        dst_shapes,
        fill_color,
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


def _behavior_stats_one(grid: np.ndarray) -> dict[str, int]:
    bg, src, _dest, src_subs, dest_holes = _extract_src_dst(grid)
    src_area_order = _sort_by_area(src_subs)
    dest_area_order = _sort_by_area(dest_holes)

    src_rank_of_comp = {id(comp): idx for idx, comp in enumerate(src_area_order)}
    dest_rank_of_hole = {id(hole): idx for idx, hole in enumerate(dest_area_order)}

    src_items = [(idx, tuple(comp["cent"])) for idx, comp in enumerate(src_subs)]
    dst_items = [(idx, tuple(hole["cent"])) for idx, hole in enumerate(dest_holes)]
    src_pos_order = _position_order([(idx, (float(r), float(c))) for idx, (r, c) in src_items])
    dst_pos_order = _position_order([(idx, (float(r), float(c))) for idx, (r, c) in dst_items])

    pos_rank_by_dst = [0] * len(dest_holes)
    for src_idx, dst_idx in zip(src_pos_order, dst_pos_order):
        pos_rank_by_dst[int(dst_idx)] = int(src_rank_of_comp[id(src_subs[int(src_idx)])])
    area_rank_by_dst = [int(dest_rank_of_hole[id(hole)]) for hole in dest_holes]

    mismatches = sum(int(a != b) for a, b in zip(pos_rank_by_dst, area_rank_by_dst))
    seed_diff = int(_seed_differs(grid, main(grid)))
    fill_color = int(src_subs[0]["color"]) if src_subs else -1
    fill_cells = int(np.count_nonzero(src == fill_color)) if src_subs else 0
    outline_cells = (
        int(np.count_nonzero((src != int(bg)) & (src != fill_color)))
        if src_subs
        else 0
    )
    return {
        "n": int(len(dest_holes)),
        "mapping_mismatch_count": int(mismatches),
        "mapping_mismatch_any": int(mismatches > 0),
        "seed_diff": int(seed_diff),
        "src_fill_cells": int(fill_cells),
        "src_outline_cells": int(outline_cells),
        "src_fill_majority": int(fill_cells > outline_cells),
    }


def behavior_report(trials: int = 100) -> dict[str, float]:
    stats = {
        "samples": 0,
        "position_vs_area_mismatch": 0,
        "seed_solver_diff": 0,
        "fill_majority": 0,
        "avg_mapping_mismatch_fraction": 0.0,
    }
    total_frac = 0.0
    for _ in range(int(trials)):
        grid = generate_input()
        info = _behavior_stats_one(grid)
        stats["samples"] += 1
        stats["position_vs_area_mismatch"] += int(info["mapping_mismatch_any"])
        stats["seed_solver_diff"] += int(info["seed_diff"])
        stats["fill_majority"] += int(info["src_fill_majority"])
        total_frac += float(info["mapping_mismatch_count"]) / float(max(1, info["n"]))

    n = float(max(1, stats["samples"]))
    return {
        "samples": float(stats["samples"]),
        "position_vs_area_mismatch_rate": float(stats["position_vs_area_mismatch"]) / n,
        "seed_solver_diff_rate": float(stats["seed_solver_diff"]) / n,
        "fill_majority_rate": float(stats["fill_majority"]) / n,
        "avg_mapping_mismatch_fraction": float(total_frac) / n,
    }
