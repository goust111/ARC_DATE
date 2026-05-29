# Combo variant of 581f7754 (index 11)
# combo_spec: D6/0+D7/0
# Name: 复制而非移动+紧凑裁切

from __future__ import annotations

# Design analysis for the D6/0 + D7/0 combo variant
# --------------------------------------------------
# a) 具体化与行为追踪
#    - 保留原题对象类型：边界孤立锚点、内部双色精灵、少数色 marker、按锚点所在边决定滑动轴。
#    - 原题流程：
#         1. 找到 marker 颜色对应的边界锚点。
#         2. 由锚点所在边决定水平/垂直滑动。
#         3. 计算平移量，把精灵移动到 marker 与锚点对齐的位置。
#         4. 清掉源位置，返回整张原尺寸画布。
#    - 本融合变体改两处：
#         1. 不清源位置，而是在原图上额外盖一个对齐副本。
#         2. 完成复制后，再把结果裁成包含所有非背景像素的最小外接矩形。
#    - 小例子：
#         * 顶边有颜色 4 的锚点，位于列 9。
#         * 某个精灵内部唯一的 4 在 (6,3)，主体颜色是 7。
#         * 原题会水平平移 +6，把整块移到 marker 列 9，旧位置清空。
#         * 本题会在旧位置保留该精灵，再在右侧列 9 处出现一份新副本。
#         * 若整张画布宽 18，但所有非背景只落在列 2..12，最终输出只返回这些列对应的紧凑窗口。
#
# b) 复杂度校准
#    - 如果允许“源副本”和“目标副本”重叠，复制效果会退化成局部加粗；因此生成器要求 source 也必须能放进 solved 的背景区。
#    - 如果同一样例里同时混用 top/bottom 与 left/right 两类锚点，tight crop 常常不明显；因此每个样例固定单一 axis family：
#         * 全是 top/bottom 锚点，此时裁切主要缩宽度；
#         * 或全是 left/right 锚点，此时裁切主要缩高度。
#    - 仍保留 3-5 个精灵、1-3 个锚点、marker 颜色可复用，避免题目退化成单对象演示。
#
# c) 隐式假设审查
#    - 锚点检测假设：边界单像素且 4 邻居全是背景。融合后仍成立，generator 保证锚点有 bg 隔离带且不在角上。
#    - marker 判定假设：真实精灵是多色连通块，少数色就是 marker。融合后仍成立，generator 继续构造“单 marker + 单 body 色”。
#    - 平移逻辑假设：锚点颜色决定匹配对象，锚点边决定滑动轴。D6/D7 都不改这一核心。
#    - 原题中 `out[coords]=bg` 依赖“移动而非复制”；融合后该前提失效，solver 必须删除清空步骤。
#    - 新增 tight crop 依赖“结果至少存在非背景像素”；锚点始终保留，因此 bbox 不会为空。
#    - 新增“不覆盖已有非背景”逻辑在复制版中更敏感；generator 必须保证所有目标副本位置在 input 中都是背景。
#
# d) 放置合法性
#    - 采用 output-first：先在 solved 画布放锚点和对齐副本，再把 source 精灵散射到 input。
#    - source 位置同时检查两件事：
#         * 在 input 中不重叠、不越界、不接触其他前景；
#         * 在 solved 中也必须仍是空背景区，这样复制时目标层和源层分离。
#    - optional singleton distractor 若启用，也按同样双层约束生成。
#
# e) 视觉可观察性
#    - 人类应能从样例直接看到两条规则：
#         * 每个精灵沿锚点指定轴再出现一份对齐副本；
#         * 输出边缘空白被统一裁掉。
#    - 复制与裁切都必须稳定可见，因此 generator 会拒绝：
#         * 裁切后尺寸没有缩小；
#         * 融合输出与“普通移动后再裁切”的结果完全相同的退化样本。
#
# f) 可视化整洁度
#    - 复制会显著增加密度，因此 source 与 solved 区域必须留出 bg 间隔，避免双层图形互相咬合。
#    - 裁切只发生在所有对象安放完成之后，不引入新的重叠，只是去掉纯背景边框。
#
# g) 变化的可推断性
#    - 不引入额外颜色映射：锚点颜色仍只用于指认 marker，body 色只负责区分对象形状。
#    - 这样解题者只需从空间现象中提炼两条新规律：复制保留源块、输出紧凑裁切。
#
# h) main() 的新增输入假设与 generator 保证
#    - 假设 1：若某单色单像素组件颜色属于锚点颜色且它自己不是锚点，则它可按同规则复制。
#      保证：distractor 仅在满足该条件时生成。
#    - 假设 2：复制目标区域若已有非背景像素，solver 会放弃覆盖。
#      保证：source 散射时显式要求 source 位置位于 solved 的背景区，避免目标冲突。
#    - 假设 3：tight crop 应该真的改变视窗，而不是偶然返回满画布。
#      保证：同一样例只用一个 axis family，并在验收时检查 crop 的对应维度确实缩小。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, find_components, get_bg, neighbors4, paste


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy interior sprites along the anchor-determined axis, then return the tight crop of the combined scene.

    Input structure:
    - The most frequent color is the background.
    - Interior foreground components are usually two-color sprites whose least-frequent color is a marker pixel.
    - Isolated non-background border singletons act as anchors, and an anchor's edge position determines the slide axis.

    Rule:
    - Match each sprite to the anchor sharing its marker color.
    - A top/bottom anchor makes the sprite copy slide horizontally so marker and anchor share a column; a left/right anchor makes it slide vertically so they share a row.
    - Keep the original sprite in place, stamp the translated copy only onto background cells, clip any out-of-bounds pixels, and finally crop the result to the minimal rectangle containing all non-background pixels.

    Key insight:
    - Color chooses which anchor a sprite follows, border side chooses the translation axis, and the output shows both the source sprite and its aligned duplicate inside a tightly cropped viewport.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    # --- STEP 1) Detect isolated border anchors. ---
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for (r, c) in anchors_by_color.values()}

    # --- STEP 2) Extract all candidate foreground components. ---
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def component_marker_color(coords: np.ndarray) -> int | None:
        vals = [int(g[int(r), int(c)]) for r, c in coords]
        cnt = Counter(vals)
        if len(cnt) <= 1:
            return None
        return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])

    def marker_pixel(coords: np.ndarray, marker: int) -> tuple[int, int] | None:
        pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0]
        cells = {(int(r), int(c)) for r, c in coords}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (int(rr), int(cc)) in cells and int(g[int(rr), int(cc)]) != int(marker):
                    return (int(r), int(c))
        return pts[0]

    moves: list[tuple[np.ndarray, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        # --- STEP 3) Determine which anchor color controls the component. ---
        marker_col = component_marker_color(coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = marker_pixel(coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]

        # --- STEP 4) Translate along the axis implied by the anchor edge. ---
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)

        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append((coords, int(dr), int(dc)))

    out = g.copy()

    # --- STEP 5) Keep the source layer and stamp translated copies. ---
    for coords, dr, dc in moves:
        for r, c in coords:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(g[int(r), int(c)])

    # --- STEP 6) Tight-crop the combined source+copy scene. ---
    return _tight_crop(out, bg)


# --------------- generator ---------------

_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (1, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 700

_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 7)
_FRAME_HEIGHT_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (3, 7)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.28
_WIDE_BODY_THRESHOLD = 0.55
_SYMMETRIC_BODY_THRESHOLD = 0.82
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.45
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_DISTRACTOR_PROB = 0.12
_DISTRACTOR_COUNT_RANGE = (1, 3)

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.append((rr, cc))
    return frontier


def _grow_body(area):
    choices = [
        (h, w)
        for h in _inclusive_range(_BODY_HEIGHT_RANGE)
        for w in _inclusive_range(_BODY_WIDTH_RANGE)
        if h * w >= area
    ]
    h, w = random.choice(choices)
    cells = set(random.choice(_GROW_BODY_SEEDS))
    while len(cells) < area:
        frontier = _frontier_cells(cells, h, w)
        if not frontier:
            return _grow_body(area)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _symmetric_body(area):
    choices = []
    for height in _inclusive_range(_BODY_HEIGHT_RANGE):
        extra = area - height
        if extra >= 2 and extra % 2 == 0 and extra // 2 <= height:
            choices.append((height, extra // 2))
    if not choices:
        return _grow_body(area)

    height, pair_count = random.choice(choices)
    cells = {(r, 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((r, 0))
        cells.add((r, 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(c, r) for r, c in cells}
    return _flip(cells)


def _wide_body(area):
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, c) for c in range(width)}
    cells.add((1, random.randrange(width)))
    while len(cells) < area:
        frontier = _frontier_cells(cells, height, width)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _frame_body(area):
    choices = []
    for h in _inclusive_range(_FRAME_HEIGHT_RANGE):
        for w in _inclusive_range(_FRAME_WIDTH_RANGE):
            perim = 2 * h + 2 * w - 4
            removable = 2 * max(0, h - 2) + 2 * max(0, w - 2)
            if perim >= area and perim - area <= min(_FRAME_MAX_GAP, removable):
                choices.append((h, w))
    if not choices:
        return _grow_body(area)

    h, w = random.choice(choices)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    gap = len(cells) - area
    if gap:
        removable = [
            (r, c)
            for r, c in cells
            if not (r in (0, h - 1) and c in (0, w - 1))
        ]
        for cell in random.sample(removable, gap):
            cells.remove(cell)
    return _flip(cells)


def _make_body(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _marker_cell(body):
    r0, c0, r1, c1 = _bbox(body)
    interior, exterior = set(), set()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            if r0 <= p[0] <= r1 and c0 <= p[1] <= c1:
                interior.add(p)
            else:
                exterior.add(p)
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg):
    body = _make_body(area)
    marker = _marker_cell(body)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas, mask, top, left, bg):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
        return False
    if top + h > H - _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_candidates(canvas, bg, axis_mode=None):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    if axis_mode in (None, "h"):
        for c in cols:
            out.append(("h", 0, c))
            out.append(("h", H - 1, c))
    if axis_mode in (None, "v"):
        for r in rows:
            out.append(("v", r, 0))
            out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions(canvas, sprite, anchor, bg):
    H, W = canvas.shape
    if anchor["axis"] == "h":
        return [
            (r, anchor["c"])
            for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
            if _can_place(canvas, sprite, r, anchor["c"], bg)
        ]
    return [
        (anchor["r"], c)
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
        if _can_place(canvas, sprite, anchor["r"], c, bg)
    ]


def _scatter_positions(inp, solved, sprite, anchor, marker, bg):
    H, W = inp.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        positions = []
        for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
            if c == mc:
                continue
            if not _can_place(inp, sprite, mr, c, bg):
                continue
            if not _can_place(solved, sprite, mr, c, bg):
                continue
            positions.append((mr, c))
        return positions

    positions = []
    for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
        if r == mr:
            continue
        if not _can_place(inp, sprite, r, mc, bg):
            continue
        if not _can_place(solved, sprite, r, mc, bg):
            continue
        positions.append((r, mc))
    return positions


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _merge_layers(base, overlay, bg):
    out = base.copy()
    overlay_mask = overlay != int(bg)
    conflict = overlay_mask & (out != int(bg)) & (overlay != out)
    if bool(conflict.any()):
        return None
    out[overlay_mask & (out == int(bg))] = overlay[overlay_mask & (out == int(bg))]
    return out


def _add_distractors(solved, inp, anchors, bg):
    if random.random() >= _DISTRACTOR_PROB:
        return True
    anchor = random.choice(anchors)
    sprite = {"patch": np.array([[int(anchor["color"])]], dtype=int), "offset": (0, 0)}
    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        pairs = []
        if anchor["axis"] == "h":
            for row in range(_INNER_BORDER_GAP, solved.shape[0] - _INNER_BORDER_GAP):
                if not _can_place(solved, sprite, row, anchor["c"], bg):
                    continue
                for in_c in range(_INNER_BORDER_GAP, solved.shape[1] - _INNER_BORDER_GAP):
                    if in_c == anchor["c"]:
                        continue
                    if not _can_place(inp, sprite, row, in_c, bg):
                        continue
                    if not _can_place(solved, sprite, row, in_c, bg):
                        continue
                    pairs.append((row, in_c))
            if not pairs:
                return False
            row, in_c = random.choice(pairs)
            _place(solved, sprite, row, anchor["c"], bg)
            _place(inp, sprite, row, in_c, bg)
            continue

        for col in range(_INNER_BORDER_GAP, solved.shape[1] - _INNER_BORDER_GAP):
            if not _can_place(solved, sprite, anchor["r"], col, bg):
                continue
            for in_r in range(_INNER_BORDER_GAP, solved.shape[0] - _INNER_BORDER_GAP):
                if in_r == anchor["r"]:
                    continue
                if not _can_place(inp, sprite, in_r, col, bg):
                    continue
                if not _can_place(solved, sprite, in_r, col, bg):
                    continue
                pairs.append((in_r, col))
        if not pairs:
            return False
        in_r, col = random.choice(pairs)
        _place(solved, sprite, anchor["r"], col, bg)
        _place(inp, sprite, in_r, col, bg)
    return True


def _combo_visibility_ok(inp, solved, expected, bg, axis_mode):
    if expected.size == 0:
        return False
    solved_crop = _tight_crop(solved, bg)
    if np.array_equal(expected, solved_crop):
        return False
    source_only = (inp != int(bg)) & (solved == int(bg))
    if not bool(source_only.any()):
        return False
    if axis_mode == "h" and int(expected.shape[1]) >= int(inp.shape[1]):
        return False
    if axis_mode == "v" and int(expected.shape[0]) >= int(inp.shape[0]):
        return False
    return True


def _gen_attempt():
    bg, solved, inp = _new_canvas_pair()
    placed = _place_anchors(solved, inp, bg)
    if placed is None:
        return None
    anchor_colors, anchors, axis_mode = placed

    marker_colors = _marker_color_sequence(anchor_colors)
    sprites = _place_aligned_sprites(solved, anchors, marker_colors, bg)
    if sprites is None:
        return None

    if not _scatter_sprites(inp, solved, sprites, bg):
        return None
    if not _add_distractors(solved, inp, anchors, bg):
        return None

    merged = _merge_layers(solved, inp, bg)
    if merged is None:
        return None
    expected = _tight_crop(merged, bg)

    if not _combo_visibility_ok(inp, solved, expected, bg, axis_mode):
        return None
    if not _generated_grid_is_valid(inp, expected, bg):
        return None
    return inp


def _new_canvas_pair():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    solved = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    return bg, solved, inp


def _place_anchors(solved, inp, bg):
    axis_mode = random.choice(("h", "v"))
    anchor_colors = random.sample([c for c in _COLORS if c != bg], random.randint(*_ANCHOR_COUNT_RANGE))
    anchors = []
    for color in anchor_colors:
        candidates = _anchor_candidates(solved, bg, axis_mode=axis_mode)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(solved, r, c, color)
        _put_anchor(inp, r, c, color)
    return anchor_colors, anchors, axis_mode


def _marker_color_sequence(anchor_colors):
    n_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    marker_colors = anchor_colors[:]
    while len(marker_colors) < n_sprites:
        marker_colors.append(random.choice(anchor_colors))
    random.shuffle(marker_colors)
    return marker_colors


def _place_aligned_sprites(solved, anchors, marker_colors, bg):
    by_color = {a["color"]: a for a in anchors}
    used_body = set()
    sprites = []
    for marker in marker_colors:
        anchor = by_color[int(marker)]
        body = _body_color(bg, marker, used_body)
        used_body.add(body)
        sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
        positions = _aligned_positions(solved, sprite, anchor, bg)
        if not positions:
            return None
        marker_pos = random.choice(positions)
        _place(solved, sprite, marker_pos[0], marker_pos[1], bg)
        sprites.append((sprite, anchor, marker_pos))
    return sprites


def _scatter_sprites(inp, solved, sprites, bg):
    random.shuffle(sprites)
    for sprite, anchor, marker_pos in sprites:
        positions = _scatter_positions(inp, solved, sprite, anchor, marker_pos, bg)
        if not positions:
            return False
        in_pos = random.choice(positions)
        _place(inp, sprite, in_pos[0], in_pos[1], bg)
    return True


def _generated_grid_is_valid(inp, out, bg):
    if not _bg_is_mode(inp, bg):
        return False
    return bool(np.array_equal(main(inp), out))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
