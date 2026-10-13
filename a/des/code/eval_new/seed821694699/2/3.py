import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

_BLACK = 0
_YELLOW = 4
_AZURE = 8


def _extract_components(grid: np.ndarray) -> Dict[int, List[np.ndarray]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: Dict[int, List[np.ndarray]] = {}
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if color == _BLACK or visited[y, x]:
                continue

            queue = [(y, x)]
            visited[y, x] = True
            pts: List[Tuple[int, int]] = []

            while queue:
                cy, cx = queue.pop()
                pts.append((cy, cx))
                for dy, dx in steps:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or int(grid[ny, nx]) != color:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

            comps.setdefault(color, []).append(np.asarray(pts, dtype=np.int16))

    return comps


def _sort_direction(grid: np.ndarray) -> bool:
    """Decode the yellow/azure sorting marker.

    Parameters
    ----------
    grid
        输入网格。

    Returns
    -------
    bool
        `True` 表示面积从小到大（顶部为 yellow），`False` 表示从大到小。

    Strategy
    --------
    在靠右侧扫描列，寻找一段由 yellow/azure 交替组成的连续竖列，取最右侧且最长的列。
    列上端颜色用于判断方向。

    Fallback
    --------
    未检测到有效列时默认小到大。
    """
    h, w = grid.shape
    best_x = -1
    best_top = _YELLOW
    best_len = 0

    for x in range(w - 1, max(-1, w - 10), -1):
        col = grid[:, x]
        ys = np.where((col == _YELLOW) | (col == _AZURE))[0]
        if ys.size < 2:
            continue

        y0, y1 = int(ys[0]), int(ys[-1])
        seg = col[y0 : y1 + 1]
        if len(seg) == 0:
            continue

        if not np.all((seg == _YELLOW) | (seg == _AZURE)):
            continue
        if np.all(seg[1:] == seg[:-1]):
            continue
        if len(seg) < 2:
            continue

        good = True
        for i in range(1, len(seg)):
            if seg[i] == seg[i - 1]:
                good = False
                break
        if not good:
            continue

        seg_len = y1 - y0 + 1
        if x > best_x or (x == best_x and seg_len > best_len):
            best_x = x
            best_len = seg_len
            best_top = int(seg[0])

    return best_top == _YELLOW


def _merge_two_fragments(a: np.ndarray, b: np.ndarray, max_shift: int) -> Tuple[int, int, int, int, np.ndarray]:
    """Find the unique translation that makes two equal-color fragments complete a full rectangle.

    Returns
    -------
    tuple
        (min_y, min_x, height, width, merged_bitmap)
    """
    a_points = np.asarray(a, dtype=np.int16)
    b_points = np.asarray(b, dtype=np.int16)

    ay = a_points[:, 0]
    ax = a_points[:, 1]
    by = b_points[:, 0]
    bx = b_points[:, 1]

    area = len(a_points) + len(b_points)
    a_set = set(map(tuple, a_points.tolist()))

    best = None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            by2 = by + dy
            bx2 = bx + dx

            overlap = False
            for y, x in zip(by2, bx2):
                if (int(y), int(x)) in a_set:
                    overlap = True
                    break
            if overlap:
                continue

            min_y = int(min(int(ay.min()), int(by2.min())))
            max_y = int(max(int(ay.max()), int(by2.max())))
            min_x = int(min(int(ax.min()), int(bx2.min())))
            max_x = int(max(int(ax.max()), int(bx2.max())))

            h = max_y - min_y + 1
            w = max_x - min_x + 1
            if h * w != area:
                continue

            # If bbox area equals combined area and overlap is zero, this is a valid reconstruction.
            merged = np.full((h, w), False, dtype=bool)
            for y, x in a_points:
                merged[y - min_y, x - min_x] = True
            for y, x in zip(by2, bx2):
                merged[y - min_y, x - min_x] = True

            if merged.all():
                if best is None:
                    best = (min_y, min_x, h, w)
                    chosen_b = np.column_stack((by2, bx2))
                    break

        if best is not None:
            break

    if best is None:
        raise ValueError("No matching translation found for paired fragments")

    min_y, min_x, h, w = best
    merged_mask = np.zeros((h, w), dtype=bool)
    for y, x in a_points:
        merged_mask[y - min_y, x - min_x] = True
    for y, x in chosen_b:
        merged_mask[int(y) - min_y, int(x) - min_x] = True

    return min_y, min_x, h, w, merged_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct every two-piece jagged rectangle, sort by area, and concatenate on one row.

    Parameters
    ----------
    grid : np.ndarray
        输入网格，使用 ARC 的整数颜色编码。

    Returns
    -------
    np.ndarray
        输出网格，黑色背景，并按面积顺序放置完整矩形（每两个拼合矩形之间留 1 列黑色间隔）。

    Algorithm
    ---------
    1. 按颜色提取 4-连通分量，过滤出恰有两个分量的颜色。
    2. 对每一对分量枚举位移，找到能形成满矩形且不重叠的拼合方式。
    3. 根据重建矩形面积排序，顺序由右下角的 yellow/azure 指示列决定。
    4. 在新画布上从左到右依次摆放，顶部对齐到第一行。

    Complexity
    ----------
    对每种颜色的两块碎片，位移枚举 O(S^2)（S<=30）且每次检查为 O(area)，
    因此在 ARC 尺寸限制下可直接接受。
    """
    g = np.asarray(grid, dtype=np.int32)
    h, w = g.shape
    comps = _extract_components(g)

    rectangles: List[Tuple[int, int, int, int, int, np.ndarray]] = []
    # (area, color, h, w, x_off, mask)

    for color, parts in comps.items():
        if color in (_BLACK, _YELLOW, _AZURE):
            continue
        if len(parts) != 2:
            continue

        try:
            min_y, min_x, rh, rw, merged_mask = _merge_two_fragments(parts[0], parts[1], max(h, w) + 1)
        except ValueError:
            continue

        rect_area = rh * rw
        rectangles.append((rect_area, color, rh, rw, 0, merged_mask))

    asc = _sort_direction(g)
    rectangles.sort(key=lambda v: v[0], reverse=not asc)

    if not rectangles:
        return np.zeros((1, 1), dtype=np.int32)

    out_h = max(r[2] for r in rectangles)
    out_w = sum(r[3] for r in rectangles) + max(0, len(rectangles) - 1)
    out = np.zeros((out_h, out_w), dtype=np.int32)

    x = 0
    for area, color, rh, rw, _, mask in rectangles:
        # mask is full rectangle filled with True (fragment union)
        rect = np.zeros((rh, rw), dtype=np.int32)
        rect[mask] = color
        out[:rh, x : x + rw] = rect
        x += rw + 1

    return out


def _split_rect(w: int, h: int, vertical: bool) -> Tuple[np.ndarray, np.ndarray]:
    part_a: List[Tuple[int, int]] = []
    part_b: List[Tuple[int, int]] = []

    if vertical:
        cut_col = random.randint(0, w - 2)
        path = [cut_col]
        for _ in range(1, h):
            cut_col = max(0, min(w - 2, cut_col + random.choice([-1, 0, 1])))
            path.append(cut_col)

        for y in range(h):
            c = path[y]
            for x in range(w):
                if x <= c:
                    part_a.append((y, x))
                else:
                    part_b.append((y, x))
    else:
        cut_row = random.randint(0, h - 2)
        path = [cut_row]
        for _ in range(1, w):
            cut_row = max(0, min(h - 2, cut_row + random.choice([-1, 0, 1])))
            path.append(cut_row)

        for x in range(w):
            r = path[x]
            for y in range(h):
                if y <= r:
                    part_a.append((y, x))
                else:
                    part_b.append((y, x))

    return np.asarray(part_a, dtype=np.int16), np.asarray(part_b, dtype=np.int16)


def generate_input() -> np.ndarray:
    """Generate one random valid-looking input instance for this puzzle format."""
    H = random.randint(14, 24)
    W = random.randint(18, 28)
    hint_col = W - 1

    grid = np.zeros((H, W), dtype=np.int32)
    used = np.zeros((H, W), dtype=bool)

    colors = [1, 2, 3, 6, 7]
    random.shuffle(colors)
    n = random.randint(2, min(4, len(colors)))
    colors = colors[:n]

    def can_place(pts: np.ndarray, ox: int, oy: int) -> bool:
        for y, x in pts:
            gy = oy + int(y)
            gx = ox + int(x)
            if gy < 0 or gy >= H or gx < 0 or gx >= hint_col:
                return False
            if used[gy, gx]:
                return False
        return True

    def paint(pts: np.ndarray, ox: int, oy: int, color: int) -> None:
        for y, x in pts:
            gy = oy + int(y)
            gx = ox + int(x)
            used[gy, gx] = True
            grid[gy, gx] = color

    for color in colors:
        w = random.randint(3, 6)
        h = random.randint(3, 6)
        w = min(w, hint_col - 1)
        h = min(h, H - 1)

        a, b = _split_rect(w, h, vertical=random.choice([True, False]))
        if a.size == 0 or b.size == 0:
            continue

        placed = False
        for _ in range(500):
            ax = random.randint(0, hint_col - w - 1)
            ay = random.randint(0, H - h)
            bx = random.randint(0, hint_col - w - 1)
            by = random.randint(0, H - h)
            if not can_place(a, ax, ay):
                continue
            if not can_place(b, bx, by):
                continue
            paint(a, ax, ay, color)
            paint(b, bx, by, color)
            placed = True
            break

        if not placed:
            # fallback to a less strict placement (same position for both if valid)
            ax = random.randint(0, hint_col - w - 1)
            ay = random.randint(0, H - h)
            for byy in range(max(0, ay - 2), min(H - h, ay + 3)):
                for bxx in range(max(0, ax - 2), min(hint_col - w, ax + 3)):
                    if can_place(a, ax, ay) and can_place(b, bxx, byy):
                        paint(a, ax, ay, color)
                        paint(b, bxx, byy, color)
                        placed = True
                        break
                if placed:
                    break

    # sorting cue: rightmost column top color indicates sort direction
    asc = random.choice([True, False])
    start_y = max(0, H - n - random.randint(0, 2))
    for i in range(n):
        if start_y + i >= H:
            break
        if asc:
            c = _YELLOW if i % 2 == 0 else _AZURE
        else:
            c = _AZURE if i % 2 == 0 else _YELLOW
        grid[start_y + i, hint_col] = c

    return grid


if __name__ == "__main__":
    inp = generate_input()
    out = main(inp)
    print(inp.shape, out.shape)
