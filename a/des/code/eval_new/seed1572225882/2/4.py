import random
from collections import deque

import numpy as np


BG = 0
COLOR_BLUE = 1
COLOR_RED = 2
COLOR_GREEN = 3
COLOR_YELLOW = 4
FRAME_COLORS = (COLOR_BLUE, COLOR_RED, COLOR_GREEN)
DEPTH_BY_COLOR = {
    COLOR_BLUE: 1,
    COLOR_RED: 2,
    COLOR_GREEN: 3,
}

def _collect_frames(grid: np.ndarray):
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    frames = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in FRAME_COLORS or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while queue:
                x, y = queue.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and grid[nx, ny] == color:
                        visited[nx, ny] = True
                        queue.append((nx, ny))

            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            if box_h < 3 or box_w < 3:
                continue

            expected = 2 * (box_h + box_w) - 4
            if len(comp) != expected:
                continue

            comp_set = set(comp)
            ok = True
            for rr in range(r0, r1 + 1):
                for cc in (c0, c1):
                    if (rr, cc) not in comp_set:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for cc in range(c0 + 1, c1):
                for rr in (r0, r1):
                    if (rr, cc) not in comp_set:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            frames.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "color": color,
                    "depth": DEPTH_BY_COLOR[color],
                    "area": (box_h - 2) * (box_w - 2),
                }
            )
    return frames


def _inside_interior(cell_r: int, cell_c: int, frame: dict):
    return frame["r0"] < cell_r < frame["r1"] and frame["c0"] < cell_c < frame["c1"]


def _inside_box(a: dict, b: dict):
    return (
        b["r0"] < a["r0"]
        and a["r1"] < b["r1"]
        and b["c0"] < a["c0"]
        and a["c1"] < b["c1"]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """从层次编码的嵌套边框内，把黄色标记按深度做分层延伸。

Input structure:
- 背景为 black。
- 仅有 1 像素宽的矩形边框：blue 为第 1 层、red 为第 2 层、green 为第 3 层。
- 每个边框内部存在若干 yellow 单像素标记，深层边框位于外层边框内部。

Rule:
- 先识别每条边框的颜色并确定其层级（blue=1, red=2, green=3）。
- 深度 1 的边框：每个 yellow 标记向右延伸成黄色水平线，直到命中其框的右内壁或内部 red 框的左壁。
- 深度 2 的边框：每个 yellow 标记向下延伸成黄色垂直线，直到命中其框的下内壁或内部 green 框的上壁。
- 深度 3 的边框：每个 yellow 标记向右下方填充黄色实心矩形，边界为该 green 框内壁。
- 按层级顺序 1→2→3 绘制，后绘制者可覆盖先绘制结果。

Key insight:
- 边框颜色不是普通装饰，而是表达嵌套层级和变换方式的控制信号；内层框同时作为外层延伸的边界与遮挡者。"""

    arr = np.array(grid, copy=True)
    if arr.ndim != 2:
        return arr

    frames = _collect_frames(arr)
    if not frames:
        return arr

    out = arr.copy()
    # 处理顺序：按深度从外到内。边框本身不变。
    for depth in (1, 2, 3):
        for frame in [f for f in frames if f["depth"] == depth]:
            children = [
                child
                for child in frames
                if child["depth"] == depth + 1 and _inside_box(child, frame)
            ]
            deep_children = [
                child
                for child in frames
                if child["depth"] > depth and _inside_box(child, frame)
            ]

            marker_cells = []
            for rr in range(frame["r0"] + 1, frame["r1"]):
                for cc in range(frame["c0"] + 1, frame["c1"]):
                    if arr[rr, cc] != COLOR_YELLOW:
                        continue
                    if any(_inside_interior(rr, cc, d) for d in deep_children):
                        continue
                    marker_cells.append((rr, cc))

            if depth == 1:
                for rr, cc in marker_cells:
                    stop = frame["c1"] - 1
                    for child in children:
                        if child["r0"] < rr < child["r1"] and child["c0"] > cc:
                            stop = min(stop, child["c0"] - 1)
                    if stop >= cc:
                        out[rr, cc : stop + 1] = COLOR_YELLOW

            elif depth == 2:
                for rr, cc in marker_cells:
                    stop = frame["r1"] - 1
                    for child in children:
                        if child["c0"] < cc < child["c1"] and child["r0"] > rr:
                            stop = min(stop, child["r0"] - 1)
                    if stop >= rr:
                        out[rr : stop + 1, cc] = COLOR_YELLOW

            else:
                if not marker_cells:
                    continue
                for rr, cc in marker_cells:
                    out[rr : frame["r1"], cc : frame["c1"]] = COLOR_YELLOW

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_inner_bbox(parent, min_h: int, min_w: int, max_h_limit: int, max_w_limit: int, rng: random.Random):
    p0, p1, p2, p3 = parent
    inner_h = p1 - p0 - 1
    inner_w = p3 - p2 - 1
    if inner_h - 2 < min_h or inner_w - 2 < min_w:
        return None

    h_hi = min(inner_h - 2, max_h_limit)
    w_hi = min(inner_w - 2, max_w_limit)
    if h_hi < min_h or w_hi < min_w:
        return None

    h = rng.randint(min_h, h_hi)
    w = rng.randint(min_w, w_hi)
    r0 = rng.randint(p0 + 1, p1 - h - 1)
    c0 = rng.randint(p2 + 1, p3 - w - 1)
    return (r0, r0 + h - 1, c0, c0 + w - 1)


def _place_bounded_box(frames, grid, bbox, color, rng):
    # Ensure no overlap / touch with existing top-level frames
    r0, r1, c0, c1 = bbox
    for old in frames:
        if not (
            r1 < old["r0"] - 1
            or r0 > old["r1"] + 1
            or c1 < old["c0"] - 1
            or c0 > old["c1"] + 1
        ):
            return False

    frames.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "color": color, "depth": DEPTH_BY_COLOR[color], "children": []})
    _draw_frame(grid, (r0, r1, c0, c1), color)
    return True


def _build_nested(grid, rng, parent, depth, max_depth=3):
    if depth > max_depth:
        return []
    if depth == 1:
        branch = rng.choice([1, 2, 3])
        if branch == 1:
            return []
        branch_three = branch == 3
    else:
        branch_three = False

    child_color = {1: COLOR_RED, 2: COLOR_GREEN}[depth]
    child_min_h = 8 if depth == 1 else 6
    child_min_w = 8 if depth == 1 else 6
    parent_area = (parent["r1"] - parent["r0"] - 1) * (parent["c1"] - parent["c0"] - 1)
    if depth == 1 and parent_area < 84:
        return []

    nested = []
    bbox = _random_inner_bbox(
        (parent["r0"], parent["r1"], parent["c0"], parent["c1"]),
        child_min_h,
        child_min_w,
        16,
        16,
        rng,
    )
    if bbox is None:
        return nested

    frame = {"r0": bbox[0], "r1": bbox[1], "c0": bbox[2], "c1": bbox[3], "color": child_color, "depth": DEPTH_BY_COLOR[child_color], "area": (bbox[1] - bbox[0] - 1) * (bbox[3] - bbox[2] - 1), "children": []}
    _draw_frame(grid, bbox, child_color)
    nested.append(frame)

    if depth == 1 and branch_three and depth < max_depth:
        nested.extend(_build_nested(grid, rng, frame, depth + 1))
    elif depth == 1 and not branch_three:
        # branch 2: 蓝框直接嵌红框，无绿框
        pass

    return nested


def _pick_marker_positions(grid: np.ndarray, frame: dict, others, rng: random.Random):
    forbidden = []
    for other in others:
        if other["depth"] <= frame["depth"]:
            continue
        if not _inside_box(other, frame):
            continue
        forbidden.append(other)

    candidates = []
    for rr in range(frame["r0"] + 1, frame["r1"]):
        for cc in range(frame["c0"] + 1, frame["c1"]):
            if any(_inside_interior(rr, cc, child) for child in forbidden):
                continue
            candidates.append((rr, cc))

    if not candidates:
        return

    count = rng.randint(1, min(4, len(candidates)))
    rrcc = rng.sample(candidates, count)
    for rr, cc in rrcc:
        grid[rr, cc] = COLOR_YELLOW


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)

    blue_frames = []
    target = rng.randint(1, 3)

    # 放置 1~3 个外层 blue frame
    for _ in range(target):
        ok = False
        for _ in range(100):
            fh = rng.randint(10, min(20, h - 4))
            fw = rng.randint(10, min(20, w - 4))
            r0 = rng.randint(0, h - fh)
            c0 = rng.randint(0, w - fw)
            bbox = (r0, r0 + fh - 1, c0, c0 + fw - 1)
            if _place_bounded_box(blue_frames, grid, bbox, COLOR_BLUE, rng):
                ok = True
                break
        if not ok:
            # 适度回退，允许少放一些外层框
            continue

        # 在某些样例里保持仅蓝框分支，另一些放入红、绿内框。
        frame = blue_frames[-1]
        nested = _build_nested(grid, rng, frame, 1)
        for f in nested:
            blue_frames.append(f)

    # 若随机失败导致无外框，给一个固定可靠样本，避免空输入导致边界退化
    if not blue_frames:
        r0 = 2
        c0 = 2
        r1 = h - 3
        c1 = w - 3
        bbox = (r0, r1, c0, c1)
        _draw_frame(grid, bbox, COLOR_BLUE)
        blue_frames.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "color": COLOR_BLUE, "depth": 1, "area": (r1 - r0 - 1) * (c1 - c0 - 1), "children": []})

    # 放置 yellow 标记
    for frame in blue_frames:
        children = [child for child in blue_frames if child["depth"] > frame["depth"] and _inside_box(child, frame)]
        _pick_marker_positions(grid, frame, children, rng)

    return grid
