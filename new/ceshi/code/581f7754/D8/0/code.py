# Variant of 581f7754
# Dimension: D8 — 多步处理：精灵移动顺序 (multi_step)
# Change: 所有精灵同时移动（先统一清除原位置，再统一粘贴到新位置），无先后顺序依赖... → 按距离排序逐个移动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original_execution:
#       - 原题先收集所有移动向量。
#       - 然后统一清空全部原位置。
#       - 最后统一把所有精灵平移到目标列/目标行，只有终点重叠才会产生影响。
#     new_execution:
#       - 仍然先根据标记色找到目标锚点，并据锚点所在边确定水平/垂直滑动轴。
#       - 但不再同时结算，而是按“标记像素到目标锚点的轴向距离”从近到远排序。
#       - 每个精灵轮到自己时，先从棋盘上拿起自身，再沿目标轴一格一格滑动。
#       - 若下一格会与现存前景重叠，或会与其发生 8 邻接贴靠，则立刻停在前一格。
#       - 因此先移动的精灵会在锚点前占位，后移动的精灵只能排队停下。
#   concrete_visualization:
#     - 多个同色精灵共享一个边界锚点，并位于同一条“车道”上：
#       top/bottom 锚点对应同一行里的水平滑动；left/right 锚点对应同一列里的垂直滑动。
#     - 输入中这些精灵离锚点的距离不同，但都朝同一方向逼近同一目标列/行。
#     - 输出中最近的精灵先到前面，后面的精灵被它挡住，形成有空隙的排队。
#   traced_example:
#     - 例：顶边锚点在列 8，三个同色精灵都在第 6 行，标记列分别是 7、4、1。
#     - 原题：三者都会直接改到标记列=8；若终点冲突，只在最终粘贴时局部缺损。
#     - 变体：列 7 的精灵先右移 1 格到目标列 8；
#       列 4 的精灵再向右滑，接近前者时因下一步会贴到已落位精灵而停下；
#       列 1 的精灵继续滑，又会被第二个精灵挡住，于是停得更远。
#     - 差异：输出不再是“所有标记都对齐到锚点列/行”，而是“按距离顺序排队逼近锚点”。
#   complexity_calibration:
#     - 排序只用可见的几何距离，禁止用颜色 id 或隐藏元数据。
#     - 生成器强制至少一条车道含 2-3 个共享同锚点的精灵，避免退化成互不干扰的原题。
#     - 仍保留 1-2 个锚点与 3-5 个精灵，视觉上与原题同族，但顺序效应稳定可见。
#   old_assumptions_that_break:
#     - 原 solver 的 `moves.append(...)` 本身没问题，问题出在后面的“clear all / paste all”。
#     - `for coords in moves: out[coords]=bg` 依赖“别的精灵不会在你旧位置上落脚”的同时结算前提。
#       变体里若仍这样做，会抹掉已经先移动到位的占位信息，顺序完全消失。
#     - 原来的最终粘贴只检查终点是否为 bg，没有任何路径概念。
#       变体里必须把“是否还能继续滑一格”放进循环，否则不会产生阻挡。
#   legality_and_generator_guarantees:
#     - 生成器只用非角落锚点，并让车道离边界至少 2 格，避免锚点本身因 8 邻接而提前阻挡。
#     - 输入中所有精灵保持 8 邻域隔离，便于 solver 正确拆分 connected components。
#     - 共享车道的精灵从锚点同一侧出发，且距离严格递增，这样“从近到远”有唯一可见顺序。
#     - 生成后额外检查：
#       `main(inp)` 不报错；
#       至少 1 个精灵被提前挡停；
#       顺序版输出与原始“同时移动版”输出不同。
#   observability:
#     - 解题者只需看 2-4 组样例就能发现：同色精灵不是都对齐到锚点，而是在锚点前排队。
#     - 排队方向由锚点边决定，队列先后由可见距离决定，没有不可见排序因素。


# Solver configuration.
_DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _detect_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = g.shape
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(r, c, H, W):
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
    return anchors_by_color


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    S = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _extract_moves(g: np.ndarray):
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors(g, bg)
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
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

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        on_top_bottom = int(ar) in (0, int(H - 1))
        on_left_right = int(ac) in (0, int(W - 1))
        horizontal = bool(on_top_bottom and not on_left_right)
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        distance = abs(int(dc) if horizontal else int(dr))
        moves.append(
            {
                "coords": coords.copy(),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "anchor": (int(ar), int(ac)),
                "horizontal": bool(horizontal),
                "dr": int(dr),
                "dc": int(dc),
                "distance": int(distance),
            }
        )
    moves.sort(key=lambda m: (int(m["distance"]), int(m["marker_pixel"][0]), int(m["marker_pixel"][1])))
    return g, bg, anchors_by_color, moves


def _can_occupy_with_gap(board: np.ndarray, coords: np.ndarray, off_r: int, off_c: int, bg: int) -> bool:
    H, W = board.shape
    translated = set()
    for r, c in coords:
        rr = int(r) + int(off_r)
        cc = int(c) + int(off_c)
        if 0 <= rr < H and 0 <= cc < W:
            translated.add((int(rr), int(cc)))

    for rr, cc in translated:
        if int(board[rr, cc]) != int(bg):
            return False

    for rr, cc in translated:
        for dr, dc in _DIR8:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (int(nr), int(nc)) in translated:
                continue
            if int(board[nr, nc]) != int(bg):
                return False
    return True


def _paste_translated(board: np.ndarray, src: np.ndarray, coords: np.ndarray, off_r: int, off_c: int) -> None:
    H, W = board.shape
    for r, c in coords:
        rr = int(r) + int(off_r)
        cc = int(c) + int(off_c)
        if 0 <= rr < H and 0 <= cc < W:
            board[rr, cc] = int(src[int(r), int(c)])


def _run_sequential(g: np.ndarray, bg: int, moves: list[dict], *, return_trace: bool = False):
    board = np.array(g, copy=True)
    trace = []

    for move in moves:
        coords = move["coords"]
        board[coords[:, 0], coords[:, 1]] = int(bg)

        step_r = 0 if int(move["dr"]) == 0 else (1 if int(move["dr"]) > 0 else -1)
        step_c = 0 if int(move["dc"]) == 0 else (1 if int(move["dc"]) > 0 else -1)
        max_steps = int(move["distance"])

        actual_steps = 0
        for k in range(1, max_steps + 1):
            if _can_occupy_with_gap(board, coords, step_r * k, step_c * k, bg):
                actual_steps = int(k)
                continue
            break

        _paste_translated(board, g, coords, step_r * actual_steps, step_c * actual_steps)
        if return_trace:
            trace.append(
                {
                    **move,
                    "actual_steps": int(actual_steps),
                    "blocked": bool(actual_steps < max_steps),
                }
            )

    if return_trace:
        return board, trace
    return board


def _run_simultaneous(g: np.ndarray, bg: int, moves: list[dict]) -> np.ndarray:
    board = np.array(g, copy=True)
    for move in moves:
        coords = move["coords"]
        board[coords[:, 0], coords[:, 1]] = int(bg)
    H, W = board.shape
    for move in moves:
        coords = move["coords"]
        for r, c in coords:
            rr = int(r) + int(move["dr"])
            cc = int(c) + int(move["dc"])
            if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == int(bg):
                board[rr, cc] = int(g[int(r), int(c)])
    return board


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g, bg, _anchors, moves = _extract_moves(grid)
    out_seq, trace = _run_sequential(g, bg, moves, return_trace=True)
    out_sim = _run_simultaneous(g, bg, moves)
    blocked = sum(1 for t in trace if t["blocked"])
    advanced = sum(1 for t in trace if int(t["actual_steps"]) > 0)
    by_anchor = Counter((int(t["anchor"][0]), int(t["anchor"][1]), bool(t["horizontal"])) for t in trace)
    return {
        "move_count": int(len(trace)),
        "blocked_count": int(blocked),
        "advanced_count": int(advanced),
        "has_conflict_lane": bool(any(v >= 2 for v in by_anchor.values())),
        "output_changed": bool(not np.array_equal(out_seq, g)),
        "changed_vs_simultaneous": bool(not np.array_equal(out_seq, out_sim)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, but resolve the motion one sprite at a time.

    Input structure:
    - The background is the most frequent color.
    - Interior connected sprites are multi-color bodies with a least-frequent marker pixel.
    - Isolated border singletons supply the anchor colors; each anchor still determines the allowed slide axis.

    Rule:
    - Match every sprite to the anchor whose color equals the sprite's marker color.
    - A top/bottom anchor requests horizontal motion; a left/right anchor requests vertical motion.
    - Sort sprites by marker-to-anchor distance along that axis from near to far, then move them sequentially.
    - On its turn, a sprite is lifted from its current spot and slides one cell at a time until it either reaches the target alignment or the next step would hit/touch existing foreground.

    Key insight:
    - The border anchor still chooses the axis, but the output now depends on motion order: early sprites claim space near the anchor and force later sprites to queue behind them.
    """
    g, bg, _anchors, moves = _extract_moves(np.array(grid, dtype=int))
    return _run_sequential(g, bg, moves)


# --------------- generator ---------------

# Canvas and palette.
_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)
_GEN_RETRY_LIMIT = 500

# Layout.
_INNER_BORDER_GAP = 1
_SAFE_LANE_MARGIN = 2
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

# Counts.
_ANCHOR_COUNT_CHOICES = (1, 2)
_SPRITE_COUNT_RANGE = (3, 5)
_PRIMARY_LANE_SIZE_RANGE = (2, 3)
_SPRITE_BUILD_RETRIES = 60
_LANE_PLACEMENT_RETRIES = 80

# Shape mix.
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BODY_AREA_RANGE = (4, 13)
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
_FLIP_PROB = 0.5

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


def _directed_marker_cell(body, axis: str, step: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(body)
    preferred = []
    fallback = []
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            fallback.append(p)
            if axis == "h":
                if step > 0 and p[1] > c1:
                    preferred.append(p)
                if step < 0 and p[1] < c0:
                    preferred.append(p)
            else:
                if step > 0 and p[0] > r1:
                    preferred.append(p)
                if step < 0 and p[0] < r0:
                    preferred.append(p)
    if preferred:
        return random.choice(list(set(preferred)))
    if fallback:
        return random.choice(list(set(fallback)))
    return None


def _make_directed_sprite(area, body_color, marker_color, bg, axis: str, step: int):
    for _ in range(_SPRITE_BUILD_RETRIES):
        body = _make_body(area)
        marker = _directed_marker_cell(body, axis, step)
        if marker is None:
            continue
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
        return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}
    return None


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


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _lane_sides(axis: str):
    return ["left", "right"] if axis == "h" else ["top", "bottom"]


def _lane_direction(axis: str, side: str) -> int:
    if axis == "h":
        return 1 if side == "left" else -1
    return 1 if side == "top" else -1


def _lane_lines(shape: tuple[int, int], axis: str):
    H, W = shape
    if axis == "h":
        return list(range(_SAFE_LANE_MARGIN, H - _SAFE_LANE_MARGIN))
    return list(range(_SAFE_LANE_MARGIN, W - _SAFE_LANE_MARGIN))


def _lane_positions(canvas, sprite, anchor, side: str, line: int, bg):
    H, W = canvas.shape
    axis = anchor["axis"]
    positions = []
    if axis == "h":
        ac = int(anchor["c"])
        if side == "left":
            cols = range(ac - 1, _INNER_BORDER_GAP - 1, -1)
        else:
            cols = range(ac + 1, W - _INNER_BORDER_GAP)
        for c in cols:
            if _can_place(canvas, sprite, int(line), int(c), bg):
                positions.append(((int(line), int(c)), abs(int(c) - ac)))
    else:
        ar = int(anchor["r"])
        if side == "top":
            rows = range(ar - 1, _INNER_BORDER_GAP - 1, -1)
        else:
            rows = range(ar + 1, H - _INNER_BORDER_GAP)
        for r in rows:
            if _can_place(canvas, sprite, int(r), int(line), bg):
                positions.append(((int(r), int(line)), abs(int(r) - ar)))
    return positions


def _place_lane(canvas, anchor, count, bg, used_body):
    axis = anchor["axis"]
    lines = _lane_lines(canvas.shape, axis)
    random.shuffle(lines)
    sides = _lane_sides(axis)
    random.shuffle(sides)

    for line in lines[:_LANE_PLACEMENT_RETRIES]:
        for side in sides:
            temp = canvas.copy()
            temp_used = set(used_body)
            placements = []
            last_distance = 0
            step = _lane_direction(axis, side)
            ok = True
            for _idx in range(count):
                placed = False
                for _ in range(_SPRITE_BUILD_RETRIES):
                    marker = int(anchor["color"])
                    body = _body_color(bg, marker, temp_used)
                    sprite = _make_directed_sprite(
                        random.randint(*_BODY_AREA_RANGE),
                        body,
                        marker,
                        bg,
                        axis,
                        step,
                    )
                    if sprite is None:
                        continue
                    candidates = [
                        (pos, dist)
                        for pos, dist in _lane_positions(temp, sprite, anchor, side, line, bg)
                        if int(dist) > int(last_distance)
                    ]
                    if not candidates:
                        continue
                    take = candidates[: min(4, len(candidates))]
                    pos, dist = random.choice(take)
                    _place(temp, sprite, pos[0], pos[1], bg)
                    temp_used.add(body)
                    placements.append(
                        {
                            "sprite": sprite,
                            "marker_pos": (int(pos[0]), int(pos[1])),
                            "body_color": int(body),
                            "distance": int(dist),
                            "side": side,
                        }
                    )
                    last_distance = int(dist)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if ok:
                canvas[:, :] = temp
                used_body |= temp_used
                return placements
    return None


def _new_canvas():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    return np.full((H, W), int(bg), dtype=int), int(bg)


def _place_anchors(canvas, bg, count):
    colors = random.sample([c for c in _COLORS if c != bg], int(count))
    anchors = []
    for color in colors:
        candidates = _anchor_candidates(canvas, bg)
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        _put_anchor(canvas, r, c, color)
        anchors.append(anchor)
    return anchors


def _gen_attempt():
    canvas, bg = _new_canvas()

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    primary_count = random.randint(_PRIMARY_LANE_SIZE_RANGE[0], min(_PRIMARY_LANE_SIZE_RANGE[1], total_sprites))
    remaining = total_sprites - primary_count
    anchor_count = 1 if remaining == 0 or random.random() < 0.6 else 2
    anchors = _place_anchors(canvas, bg, random.choice([n for n in _ANCHOR_COUNT_CHOICES if n <= anchor_count]))
    if anchors is None:
        return None

    used_body = set()
    primary_anchor = random.choice(anchors)
    if _place_lane(canvas, primary_anchor, primary_count, bg, used_body) is None:
        return None

    extra_anchors = anchors[:]
    for _ in range(remaining):
        anchor = random.choice(extra_anchors)
        if _place_lane(canvas, anchor, 1, bg, used_body) is None:
            return None

    if not _bg_is_mode(canvas, bg):
        return None

    metrics = _variant_metrics(canvas)
    if int(metrics["move_count"]) < total_sprites:
        return None
    if not bool(metrics["has_conflict_lane"]):
        return None
    if int(metrics["blocked_count"]) < 1:
        return None
    if int(metrics["advanced_count"]) < 1:
        return None
    if not bool(metrics["output_changed"]):
        return None
    if not bool(metrics["changed_vs_simultaneous"]):
        return None
    _ = main(canvas)
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 D8 input")
