# Combo variant of 247ef758 (index 30)
# combo_spec: D3/0+D8/2
# Name: 水平分隔+内部标记

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for combo D3/0 + D8/2
# ---------------------------------------------------------------------------
#
# a) 具体化与行为追踪
# - 原题输入是 `左侧精灵区 | 竖直分隔列 | 右侧框架`。框架边缘携带彩色标记：
#   左右边同色定义目标行，上下边同色定义目标列，最后在这些行列的笛卡尔积中心
#   印刻同色精灵。
# - 本融合版同时改两处：
#   1. D8/2: 布局改成 `上方精灵区 / 水平分隔行 / 下方框架`。
#   2. D3/0: 框架边缘恢复成统一底色，不再带信息；信息改为框内一条引导行和一条
#      引导列上的内部彩色点。若颜色 X 在 `(r, guide_col)` 出现，则 r 是 X 的目标行；
#      若颜色 X 在 `(guide_row, c)` 出现，则 c 是 X 的目标列。
# - 小例子：某个顶部精灵颜色 4，如果框内 `guide_col` 的第 9 行有一个 4 点，且
#   `guide_row` 的第 12 列有一个 4 点，则目标中心是 `(9, 12)`；若同色还有第 14 列点，
#   就会在 `(9,12)` 和 `(9,14)` 都印刻。
# - 差异总结：原题“看边框、左右布局”；融合版“看框内点、上下布局”。精灵移动与覆盖
#   规则本身不变，但解题者必须先识别完全不同的空间组织，再在框内部而非边缘寻找标记。
#
# b) 复杂度校准
# - 仍保留 2-3 个 mover 颜色，每个颜色有 1-2 个行标记、1-2 个列标记，保证样例中
#   会出现重复证据，而不是一次性映射。
# - 内部标记采用“点”而不是整条线或厚十字，视觉上更干净，也不会把框内背景切碎。
# - 下方框架横向铺满，顶部精灵横向打包，能明显区分于原题的左右布局，避免视觉退化。
#
# c) 隐式假设审查
# - `largest_enclosed_component_bbox(g == bg)` 依赖“框内部背景块是最大且不接触外边界的
#   背景连通块”。本融合版中，内部标记只是少量点，不会在 4 连通下切断内部背景，前提仍成立。
# - 原题按列扫描 divider，并默认 mover 都在 `fc0` 左边；D8/2 破坏了这个前提，因此 solver
#   必须改为按行扫描 divider，并在 `fr0` 上方提取精灵。
# - 原题的目标提取依赖“框边缘带颜色”；D3/0 破坏了这个前提，因此 solver 必须改为读取
#   `guide_row/guide_col` 上的内部点。
# - 叠加两项改动后，新前提是：顶部区域只有精灵；分隔行是唯一近乎整行同色的非背景行；
#   框边缘始终统一底色；内部引导行/列上只有 mover 颜色点。generator 需要显式保证这些条件。
#
# d) 放置合法性
# - 顶部 strip 高度固定为 3-4，精灵高度不超过 strip。
# - 为避免输出印刻覆盖内部标记，引导行/列固定靠近框内左上角，而目标中心只允许选在其
#   下方/右方并留出至少 sprite 半径的安全边距。
# - 不同 mover 不共享目标行或目标列，避免歧义；所有精灵在顶部横向打包，不重叠不越界。
#
# e) 视觉可观察性
# - 人类从 2-4 个样例中可以直接看出：分隔从竖直变成水平，精灵来源区跑到上方，框边缘不再
#   带彩色信息，而框内部出现彩色点。两处变化都是强视觉信号。
#
# f) 可视化整洁度
# - 顶部只放精灵，底部只放框和少量内部点，中间是一整行纯色 divider，区域职责清晰。
# - 内部点集中在单条引导行/列上，数量受控，不会让框内变成噪声。
#
# g) 变化的可推断性
# - 颜色语义保持不变：精灵颜色 = 标记颜色 = 输出印刻颜色。没有额外颜色重映射。
# - 由于每个 mover 至少拥有一行一列标记，且同色会在输入顶部与框内重复出现，人类可从少量
#   样例稳定归纳“同色匹配后在交叉点复制”这一规则。
#
# New solver assumptions introduced here and guaranteed by the generator:
# - 恰有一个水平 divider 行位于框架上方，并且几乎整行都是同一种非背景色。
# - 框架横向铺满下半区，边框为单一 frame_base 颜色，没有边缘彩色标记。
# - 内部 `guide_row` 与 `guide_col` 在框内存在，且仅包含 mover 颜色的标记点。
# - 所有目标中心都离 `guide_row` / `guide_col` 足够远，使输出印刻不会覆盖这些标记点。


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does NOT touch the grid border.
    """
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
        if bool(
            (rs == 0).any()
            or (rs == int(h - 1)).any()
            or (cs == 0).any()
            or (cs == int(w - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - sprites live in a top strip
    - a uniform horizontal divider separates them from the lower frame
    - the frame border is uniform; internal dots on a guide row/column encode
      the target rows and columns for each sprite color
    - movers are erased from the top and stamped at every row-dot × col-dot
      intersection, with lower source sprites stamped first
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for r in range(0, max(0, int(fr0))):
        rowv = g[int(r), :]
        non = rowv[rowv != int(bg)]
        if int(len(non)) >= int(w - 1) and len(set(map(int, non))) == 1:
            divider_mask[int(r), :] = True

    top_rows = np.arange(h)[:, None] < int(fr0)

    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & top_rows & (~divider_mask)
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
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    assert guide_row <= int(ir1) and guide_col <= int(ic1)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        val = int(g[int(r), int(guide_col)])
        if val != int(bg) and val in move_colors:
            row_marks[int(val)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        val = int(g[int(guide_row), int(c)])
        if val != int(bg) and val in move_colors:
            col_marks[int(val)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {
            (int(r), int(c))
            for r in row_marks[int(col)]
            for c in col_marks[int(col)]
        }
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(
        key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]))
    )

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [
            (int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2))
            for rr, cc2 in np.argwhere(s["mask"])
        ]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


def _pick_marks(avail: list[int], preferred_n: int, min_spacing: int) -> list[int] | None:
    marks = _pick_spaced(list(avail), preferred_n, min_spacing)
    if marks or preferred_n == 1:
        return marks
    return _pick_spaced(list(avail), 1, min_spacing)


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 4]), max_h)
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


def _pack_top_items(
    items: list[tuple[int, np.ndarray, bool]],
    strip_h: int,
    width: int,
) -> list[tuple[int, np.ndarray, bool, int, int]] | None:
    """
    Pack sprites left-to-right in the top strip.
    Small vertical offsets keep bottom-to-top source ordering observable.
    """
    widths = [int(sp.shape[1]) for _, sp, _ in items]
    min_required = sum(widths) + max(0, len(items) - 1)
    if min_required > width:
        return None

    placed: list[tuple[int, np.ndarray, bool, int, int]] = []
    cursor = 0
    for idx, (col, sp, is_distractor) in enumerate(items):
        sh, sw = map(int, sp.shape)
        if sh > strip_h:
            return None
        remain_width = sum(widths[idx + 1 :]) + max(0, len(items) - idx - 2)
        max_start = int(width - sw - remain_width)
        if cursor > max_start:
            return None
        c_off = random.randint(cursor, max_start)
        r_off = random.randint(0, strip_h - sh)
        placed.append((int(col), sp, bool(is_distractor), int(r_off), int(c_off)))
        cursor = int(c_off + sw + 1)
    return placed


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    w = random.randint(14, 20)
    sprite_area_h = random.randint(3, 4)
    interior_h = random.randint(6, 8)
    h = sprite_area_h + 1 + interior_h + 2
    if not 10 <= h <= 15:
        return None

    div_row = int(sprite_area_h)
    fr0, fr1 = int(div_row + 1), int(h - 1)
    fc0, fc1 = 0, int(w - 1)
    ir0, ic0 = int(fr0 + 1), 1
    ir1, ic1 = int(h - 2), int(w - 2)
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    if guide_row > ir1 or guide_col > ic1:
        return None

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = 2 if interior_h <= 6 else random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sprite_max_w = 4 if w <= 16 else 5

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, sprite_area_h, sprite_max_w)
        if s is None:
            return None
        sprites.append(s)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = map(int, sp.shape)
        top_extent = int(sh // 2)
        bottom_extent = int(sh - 1 - top_extent)
        left_extent = int(sw // 2)
        right_extent = int(sw - 1 - left_extent)

        min_row = int(guide_row + top_extent + 1)
        max_row = int(ir1 - bottom_extent)
        min_col = int(guide_col + left_extent + 1)
        max_col = int(ic1 - right_extent)
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        desired_rows = 2 if random.random() < 0.4 else 1
        row_marks = _pick_marks(avail_rows, desired_rows, max(2, sh))
        if not row_marks:
            return None

        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        desired_cols = 2 if random.random() < 0.6 else 1
        col_marks = _pick_marks(avail_cols, desired_cols, max(2, sw))
        if not col_marks:
            return None

        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[int(div_row), :] = int(divider_color)

    inp[int(fr0), :] = int(frame_base)
    inp[int(fr1), :] = int(frame_base)
    inp[int(fr0) : int(fr1) + 1, int(fc0)] = int(frame_base)
    inp[int(fr0) : int(fr1) + 1, int(fc1)] = int(frame_base)

    for mc, (rmarks, cmarks) in mover_targets.items():
        for r in rmarks:
            inp[int(r), int(guide_col)] = int(mc)
        for c in cmarks:
            inp[int(guide_row), int(c)] = int(mc)

    items: list[tuple[int, np.ndarray, bool]] = [
        (int(mc), sp, False) for mc, sp in zip(mover_colors, sprites)
    ]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), sprite_area_h, sprite_max_w)
        if ds is None:
            return None
        items.append((int(dc_color), ds, True))

    placed = _pack_top_items(items, sprite_area_h, w)
    if placed is None:
        return None

    for col, sp, _is_distractor, r_off, c_off in placed:
        sh, sw = map(int, sp.shape)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != bg:
                    inp[int(r_off + dr), int(c_off + dc)] = int(col)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
