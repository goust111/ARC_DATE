# Combo variant of 581f7754 (index 3)
# combo_spec: D5/1+D8/0
# Name: 碰撞停止+按距离排序移动

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# combo:
#   concrete_behavior:
#     - 仍然是“内部双色/多色 sprite + 边界孤立 anchor”的原题骨架，marker 色匹配 anchor 色，anchor 边决定水平/垂直滑动轴。
#     - 但求解不再同时结算，也不再直接一次性平移到目标列/行；而是先按 marker 到 anchor 的轴向距离从近到远排序。
#     - 每个 sprite 轮到自己时，只临时拿走自己，其他未移动 sprite 仍留在棋盘上；随后沿唯一轴逐格试走。
#     - 下一步只要有任一像素会越界或撞到任一非背景像素，就立刻停在前一格。于是“先移动者占位，后移动者排队”与
#       “逐格碰撞停止”是同一条因果链，而不是两个互不相关的补丁。
#   behavior_trace:
#     - 小例子：顶边 anchor 在列 9，同一行上有三个同色 sprite，marker 列分别是 8、5、2。
#     - 原题：三者都直接把 marker 对齐到列 9，统一清空再统一 paste，路径中发生什么不重要。
#     - 融合后：列 8 的 sprite 先右移 1 格到目标列 9；列 5 的 sprite 再逐格右移，直到下一步会撞到第一个 sprite，
#       因而停在其后；列 2 的 sprite 再移动，又会被第二个 sprite 挡住。输出表现成朝 anchor 排队，而不是全部对齐。
#     - 若改回“同时移动但有碰撞停止”，因为大家彼此都被先清空，后两个 sprite 又会一路到 anchor 线；若改回
#       “顺序移动但末端不看路径”，则不会出现中途截停。必须两条改动同时存在，排队现象才成立。
#   complexity_calibration:
#     - generator 强制至少 1 条主车道包含 2-3 个共享同 anchor 的 sprite，避免退化成彼此独立的原题。
#     - 主车道复用同一几何模板但用不同 body 色，让人类肉眼能看出它们会走同一路径，也能区分不同个体。
#     - 额外 sprite 数量受控在 0-2 个，只作背景复杂度，不引入新的排序规则；排序只依赖可见的轴向距离。
#   main_assumption_audit:
#     - 原版 `clear all movers first` 的前提是“别的 sprite 的当前位置不会影响你的路径”；融合后失效，因此改成
#       “只清当前 sprite”。
#     - 原版 `paste at total (dr,dc)` 假设中间路径无关；融合后必须逐步检查每一格，否则不会产生中途停下。
#     - 原版允许越界裁剪；融合后若下一步越界，应停在边界前一格，所以 generator 也不依赖裁剪制造样例。
#     - 新 solver 额外假设：同一关键冲突车道里的距离顺序最好唯一可见。generator 用严格递增的轴向距离保证这一点。
#   generator_legality:
#     - 输入里所有 sprite 在初始态保持 8 邻域间隔，确保 connected-components 稳定。
#     - anchor 颜色互异且不用于 body；这样 marker 配对和 anchor 检测都无歧义。
#     - 主车道上的重复几何 sprite 共享同一行/列与同一侧出发方向，因而确实会争抢同一终点路径。
#     - 生成后显式检查：至少 1 个 sprite 被先移动的 sprite 挡停，且组合版输出不同于原始同时移动版。
#   observability:
#     - 人类从 2-3 组样例里能看到：离 anchor 最近的那个先贴近 anchor，后面的同色 sprite 只能一串排在后面。
#     - 排序依据是可见的距离，不使用颜色编号或隐藏元数据；body 颜色只区分个体。
#   cleanliness:
#     - 不额外加入静态障碍物，避免把“被前车挡住”和“被杂物挡住”混在一起。
#     - 主车道外的附加 sprite 只做少量点缀，避免同一区域叠太多层导致输出难读。


# Shared palette / canvas configuration.
_COLORS = tuple(range(10))
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_INNER_BORDER_GAP = 1
_SAFE_LANE_MARGIN = 2
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_ANCHOR_COUNT_CHOICES = (1, 2)
_SPRITE_COUNT_RANGE = (3, 5)
_PRIMARY_LANE_SIZE_RANGE = (2, 3)
_GEN_RETRY_LIMIT = 500
_SPRITE_BUILD_RETRIES = 70
_LANE_PLACEMENT_RETRIES = 80

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


def _directed_marker_cell(body, axis: str, step: int):
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
    choices = preferred if preferred else fallback
    if not choices:
        return None
    return random.choice(list(set(choices)))


def _make_directed_geometry(area: int, axis: str, step: int):
    for _ in range(_SPRITE_BUILD_RETRIES):
        body = _make_body(area)
        marker = _directed_marker_cell(body, axis, step)
        if marker is None:
            continue
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        body_norm = sorted((r - r0, c - c0) for r, c in body)
        marker_norm = (marker[0] - r0, marker[1] - c0)
        return {
            "body": body_norm,
            "marker": marker_norm,
            "shape": (r1 - r0 + 1, c1 - c0 + 1),
            "offset": marker_norm,
        }
    return None


def _sprite_from_geometry(geom, body_color: int, marker_color: int, bg: int):
    h, w = geom["shape"]
    patch = np.full((int(h), int(w)), int(bg), dtype=int)
    for r, c in geom["body"]:
        patch[int(r), int(c)] = int(body_color)
    mr, mc = geom["marker"]
    patch[int(mr), int(mc)] = int(marker_color)
    return {"patch": patch, "offset": (int(geom["offset"][0]), int(geom["offset"][1]))}


def _top_left(sprite, marker_r: int, marker_c: int):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas, mask, top: int, left: int, bg: int):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas, sprite, marker_r: int, marker_c: int, bg: int):
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


def _place(canvas, sprite, marker_r: int, marker_c: int, bg: int):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], int(top), int(left), transparent=int(bg))


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _detect_anchors(g: np.ndarray, bg: int):
    H, W = g.shape
    anchors_by_color = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), int(H), int(W)):
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


def _component_marker_color(g: np.ndarray, coords: np.ndarray):
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int):
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


def _extract_movers(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors(g, int(bg))
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    movers = []
    for comp_index, comp in enumerate(comps):
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
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        distance = abs(int(dc) if horizontal else int(dr))
        movers.append(
            {
                "id": int(comp_index),
                "coords": coords.astype(int),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "anchor": (int(ar), int(ac)),
                "horizontal": bool(horizontal),
                "dr": int(dr),
                "dc": int(dc),
                "distance": int(distance),
            }
        )

    movers.sort(key=lambda m: (int(m["distance"]), int(m["marker_pixel"][0]), int(m["marker_pixel"][1])))
    return g, bg, anchors_by_color, movers


def _translated_cells(coords: np.ndarray, off_r: int, off_c: int):
    return [(int(r) + int(off_r), int(c) + int(off_c)) for r, c in coords]


def _run_combo(g: np.ndarray, bg: int, movers: list[dict], *, return_trace: bool = False):
    board = np.array(g, copy=True)
    trace = []
    owner = None
    moved_ids = set()
    if return_trace:
        owner = np.full(board.shape, -1, dtype=int)
        for move in movers:
            coords = move["coords"]
            owner[coords[:, 0], coords[:, 1]] = int(move["id"])
        for move in movers:
            for r, c in move["coords"]:
                board[int(r), int(c)] = int(g[int(r), int(c)])

    for move in movers:
        coords = move["coords"]
        src_vals = [int(g[int(r), int(c)]) for r, c in coords]
        board[coords[:, 0], coords[:, 1]] = int(bg)
        if owner is not None:
            owner[coords[:, 0], coords[:, 1]] = -1

        step_r = 0 if int(move["dr"]) == 0 else (1 if int(move["dr"]) > 0 else -1)
        step_c = 0 if int(move["dc"]) == 0 else (1 if int(move["dc"]) > 0 else -1)
        distance = int(move["distance"])

        actual_steps = 0
        blocker_ids = set()
        blocked_by_wall = False
        while actual_steps < distance:
            blocked = False
            nxt_cells = _translated_cells(coords, step_r * (actual_steps + 1), step_c * (actual_steps + 1))
            for rr, cc in nxt_cells:
                if not (0 <= rr < board.shape[0] and 0 <= cc < board.shape[1]):
                    blocked = True
                    blocked_by_wall = True
                    break
                if int(board[int(rr), int(cc)]) != int(bg):
                    blocked = True
                    if owner is not None:
                        blocker_ids.add(int(owner[int(rr), int(cc)]))
                    break
            if blocked:
                break
            actual_steps += 1

        for (r, c), val in zip(coords, src_vals):
            rr = int(r) + step_r * actual_steps
            cc = int(c) + step_c * actual_steps
            board[int(rr), int(cc)] = int(val)
            if owner is not None:
                owner[int(rr), int(cc)] = int(move["id"])

        if return_trace:
            blocker_ids.discard(-1)
            trace.append(
                {
                    **move,
                    "actual_steps": int(actual_steps),
                    "blocked": bool(actual_steps < distance),
                    "blocked_by_prior": bool(any(int(bid) in moved_ids for bid in blocker_ids)),
                    "blocked_by_unmoved": bool(any(int(bid) not in moved_ids for bid in blocker_ids)),
                    "blocked_by_wall": bool(blocked_by_wall),
                }
            )
            moved_ids.add(int(move["id"]))

    if return_trace:
        return board, trace
    return board


def _run_original_simultaneous(g: np.ndarray, bg: int, movers: list[dict]):
    board = np.array(g, copy=True)
    H, W = board.shape
    for move in movers:
        coords = move["coords"]
        board[coords[:, 0], coords[:, 1]] = int(bg)
    for move in movers:
        for r, c in move["coords"]:
            rr = int(r) + int(move["dr"])
            cc = int(c) + int(move["dc"])
            if 0 <= rr < H and 0 <= cc < W and int(board[int(rr), int(cc)]) == int(bg):
                board[int(rr), int(cc)] = int(g[int(r), int(c)])
    return board


def _run_collision_without_order(g: np.ndarray, bg: int, movers: list[dict]):
    static = np.array(g, copy=True)
    for move in movers:
        coords = move["coords"]
        static[coords[:, 0], coords[:, 1]] = int(bg)

    board = np.array(static, copy=True)
    for move in movers:
        coords = move["coords"]
        src_vals = [int(g[int(r), int(c)]) for r, c in coords]
        step_r = 0 if int(move["dr"]) == 0 else (1 if int(move["dr"]) > 0 else -1)
        step_c = 0 if int(move["dc"]) == 0 else (1 if int(move["dc"]) > 0 else -1)
        distance = int(move["distance"])
        actual_steps = 0
        while actual_steps < distance:
            blocked = False
            for rr, cc in _translated_cells(coords, step_r * (actual_steps + 1), step_c * (actual_steps + 1)):
                if not (0 <= rr < board.shape[0] and 0 <= cc < board.shape[1]):
                    blocked = True
                    break
                if int(static[int(rr), int(cc)]) != int(bg):
                    blocked = True
                    break
            if blocked:
                break
            actual_steps += 1
        for (r, c), val in zip(coords, src_vals):
            rr = int(r) + step_r * actual_steps
            cc = int(c) + step_c * actual_steps
            board[int(rr), int(cc)] = int(val)
    return board


def _variant_metrics(grid: np.ndarray):
    g, bg, _anchors, movers = _extract_movers(grid)
    out_combo, trace = _run_combo(g, bg, movers, return_trace=True)
    out_sim = _run_original_simultaneous(g, bg, movers)
    out_static = _run_collision_without_order(g, bg, movers)
    by_anchor = Counter((int(t["anchor"][0]), int(t["anchor"][1]), bool(t["horizontal"])) for t in trace)
    return {
        "move_count": int(len(trace)),
        "blocked_count": int(sum(1 for t in trace if t["blocked"])),
        "blocked_by_prior": int(sum(1 for t in trace if t["blocked_by_prior"])),
        "blocked_by_unmoved": int(sum(1 for t in trace if t["blocked_by_unmoved"])),
        "has_conflict_lane": bool(any(v >= 2 for v in by_anchor.values())),
        "output_changed": bool(not np.array_equal(out_combo, g)),
        "changed_vs_original": bool(not np.array_equal(out_combo, out_sim)),
        "changed_vs_static_collision": bool(not np.array_equal(out_combo, out_static)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, resolving them in distance order with collision stopping.

    Input structure:
    - The background is the most frequent color.
    - Interior foreground components are mostly multi-color sprites whose least-frequent marker color matches a border anchor.
    - Isolated non-background border singletons are anchors; top/bottom anchors define horizontal lanes, left/right anchors define vertical lanes.

    Rule:
    - Pair each movable sprite with the anchor of the same marker color.
    - Measure the sprite's marker-to-anchor distance along that anchor's axis and process sprites from near to far.
    - On its turn, remove only that sprite from the board, then slide it one cell at a time toward the target column or row.
    - If the next step would leave the grid or hit any non-background pixel, stop immediately; otherwise continue until the target alignment is reached.

    Key insight:
    - The anchor still chooses the motion axis, but the final state now depends on both path collisions and move order: early sprites settle first and become the blockers that stop later sprites.
    """
    g, bg, _anchors, movers = _extract_movers(np.array(grid, dtype=int))
    return _run_combo(g, bg, movers)


def _isolated_bg_cell(canvas, r: int, c: int, bg: int):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, int(r) - radius), min(H, int(r) + radius + 1)
    c0, c1 = max(0, int(c) - radius), min(W, int(c) + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r: int, c: int, bg: int):
    H, W = canvas.shape
    if not (int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))):
        return False
    if int(r) in (0, int(H - 1)) and int(c) in (0, int(W - 1)):
        return False
    return _isolated_bg_cell(canvas, int(r), int(c), int(bg))


def _anchor_candidates(canvas, bg: int):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, int(r), int(c), int(bg))]


def _put_anchor(canvas, r: int, c: int, color: int):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _lane_sides(axis: str):
    return ["left", "right"] if axis == "h" else ["top", "bottom"]


def _lane_direction(axis: str, side: str):
    if axis == "h":
        return 1 if side == "left" else -1
    return 1 if side == "top" else -1


def _lane_lines(shape: tuple[int, int], axis: str):
    H, W = shape
    if axis == "h":
        return list(range(_SAFE_LANE_MARGIN, H - _SAFE_LANE_MARGIN))
    return list(range(_SAFE_LANE_MARGIN, W - _SAFE_LANE_MARGIN))


def _lane_positions(canvas, sprite, anchor, side: str, line: int, bg: int):
    H, W = canvas.shape
    axis = anchor["axis"]
    positions = []
    if axis == "h":
        ac = int(anchor["c"])
        cols = range(ac - 1, _INNER_BORDER_GAP - 1, -1) if side == "left" else range(ac + 1, W - _INNER_BORDER_GAP)
        for c in cols:
            if _can_place(canvas, sprite, int(line), int(c), bg):
                positions.append(((int(line), int(c)), abs(int(c) - ac)))
    else:
        ar = int(anchor["r"])
        rows = range(ar - 1, _INNER_BORDER_GAP - 1, -1) if side == "top" else range(ar + 1, H - _INNER_BORDER_GAP)
        for r in rows:
            if _can_place(canvas, sprite, int(r), int(line), bg):
                positions.append(((int(r), int(line)), abs(int(r) - ar)))
    return positions


def _body_color(bg: int, anchor_colors, used_body):
    pool = [c for c in _COLORS if c not in set(anchor_colors) | {int(bg)} and c not in used_body]
    if not pool:
        pool = [c for c in _COLORS if c not in set(anchor_colors) | {int(bg)}]
    return int(random.choice(pool))


def _bg_is_mode(grid: np.ndarray, bg: int):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    return np.full((H, W), int(bg), dtype=int), int(bg)


def _place_anchors(canvas, bg: int, count: int):
    colors = random.sample([c for c in _COLORS if c != int(bg)], int(count))
    anchors = []
    for color in colors:
        candidates = _anchor_candidates(canvas, int(bg))
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        _put_anchor(canvas, int(r), int(c), int(color))
        anchors.append(anchor)
    return anchors


def _place_primary_lane(canvas, anchor, count: int, bg: int, anchor_colors, used_body):
    axis = anchor["axis"]
    lines = _lane_lines(canvas.shape, axis)
    random.shuffle(lines)
    sides = _lane_sides(axis)
    random.shuffle(sides)

    for line in lines[:_LANE_PLACEMENT_RETRIES]:
        for side in sides:
            step = _lane_direction(axis, side)
            geom = _make_directed_geometry(random.randint(*_BODY_AREA_RANGE), axis, step)
            if geom is None:
                continue

            temp = canvas.copy()
            placements = []
            last_distance = 0
            ok = True
            lane_used_body = set()
            for _ in range(count):
                body = _body_color(bg, anchor_colors, used_body | lane_used_body)
                sprite = _sprite_from_geometry(geom, int(body), int(anchor["color"]), int(bg))
                candidates = [
                    (pos, dist)
                    for pos, dist in _lane_positions(temp, sprite, anchor, side, line, bg)
                    if int(dist) > int(last_distance)
                ]
                if not candidates:
                    ok = False
                    break
                near_choices = candidates[: min(5, len(candidates))]
                pos, dist = random.choice(near_choices)
                _place(temp, sprite, pos[0], pos[1], bg)
                placements.append({"sprite": sprite, "marker_pos": pos, "distance": int(dist), "line": int(line), "side": side})
                lane_used_body.add(int(body))
                last_distance = int(dist)
            if ok:
                canvas[:, :] = temp
                used_body.update(lane_used_body)
                return placements
    return None


def _place_extra_sprite(canvas, anchors, bg: int, anchor_colors, used_body):
    anchor_order = anchors[:]
    random.shuffle(anchor_order)
    for anchor in anchor_order:
        axis = anchor["axis"]
        lines = _lane_lines(canvas.shape, axis)
        random.shuffle(lines)
        sides = _lane_sides(axis)
        random.shuffle(sides)
        for line in lines[:_LANE_PLACEMENT_RETRIES]:
            for side in sides:
                step = _lane_direction(axis, side)
                geom = _make_directed_geometry(random.randint(*_BODY_AREA_RANGE), axis, step)
                if geom is None:
                    continue
                body = _body_color(bg, anchor_colors, used_body)
                sprite = _sprite_from_geometry(geom, int(body), int(anchor["color"]), int(bg))
                candidates = _lane_positions(canvas, sprite, anchor, side, line, bg)
                if not candidates:
                    continue
                pos, _dist = random.choice(candidates)
                _place(canvas, sprite, pos[0], pos[1], bg)
                used_body.add(int(body))
                return True
    return False


def _gen_attempt():
    canvas, bg = _new_canvas()
    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    primary_count = random.randint(_PRIMARY_LANE_SIZE_RANGE[0], min(_PRIMARY_LANE_SIZE_RANGE[1], total_sprites))
    extra_count = total_sprites - primary_count
    anchor_count = random.choice(_ANCHOR_COUNT_CHOICES)
    anchors = _place_anchors(canvas, bg, anchor_count)
    if anchors is None:
        return None

    used_body = set()
    anchor_colors = [int(a["color"]) for a in anchors]
    primary_anchor = random.choice(anchors)
    primary_meta = _place_primary_lane(canvas, primary_anchor, int(primary_count), int(bg), anchor_colors, used_body)
    if primary_meta is None:
        return None

    for _ in range(extra_count):
        if not _place_extra_sprite(canvas, anchors, int(bg), anchor_colors, used_body):
            return None

    if not _bg_is_mode(canvas, int(bg)):
        return None

    metrics = _variant_metrics(canvas)
    if int(metrics["move_count"]) != total_sprites:
        return None
    if not bool(metrics["has_conflict_lane"]):
        return None
    if int(metrics["blocked_by_prior"]) < 1:
        return None
    if not bool(metrics["changed_vs_original"]):
        return None
    if not bool(metrics["changed_vs_static_collision"]):
        return None
    _ = main(canvas)
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
