import numpy as np
from collections import deque
from typing import Dict, List, Tuple


def _infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    wall = None
    best = -1
    for v, c in zip(values, counts):
        if int(v) == bg:
            continue
        if int(c) > best:
            best = int(c)
            wall = int(v)
    if wall is None:
        return bg
    return wall


def _connected_components(mask: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for sr in range(h):
        row = mask[sr]
        for sc in range(w):
            if not row[sc] or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            coords = []
            touches_edge = False
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            components.append((np.array(coords, dtype=np.int16), touches_edge))
    return components


def _extract_templates(grid: np.ndarray, bg: int) -> Dict[int, np.ndarray]:
    h, w = grid.shape
    patterns: Dict[int, np.ndarray] = {}
    for mr in range(0, h - 3):
        for mc in range(1, w - 1):
            tr, tc = mr + 1, mc - 1
            patch = grid[tr : tr + 3, tc : tc + 3]
            non_bg = [v for v in np.unique(patch) if int(v) != bg]
            if len(non_bg) != 1:
                continue
            color = int(non_bg[0])
            if color == int(grid[mr, mc]):
                continue
            marker_ok = True
            if color == bg:
                marker_ok = False
            if not marker_ok:
                continue
            pattern = (patch == color)
            if not np.any(pattern):
                continue
            # keep the first detected placement for this color
            if color not in patterns:
                patterns[color] = pattern.astype(bool)
    return patterns


def _seed_direction(seed: Tuple[int, int], center_r: float, center_c: float) -> str:
    sr, sc = seed
    if abs(sr - center_r) >= abs(sc - center_c):
        return "up" if sr <= center_r else "down"
    return "left" if sc <= center_c else "right"


def _rotate_pattern(pattern: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return pattern
    if direction == "right":
        return np.rot90(pattern, k=-1)
    if direction == "down":
        return np.rot90(pattern, k=2)
    return np.rot90(pattern, k=1)


def main(grid: np.ndarray) -> np.ndarray:
    """按种子颜色选择模板，并按种子相对区域中心方位填充闭合区域内部。
Input structure:
- bg 为常见颜色；若干被灰色围线分隔的内部区域包含一个非 bg 彩色种子。
- 网格边缘上有若干 3×3 图案模板，每个模板用单一颜色绘制，并有一个 Yellow 像素在其正上方表示默认朝上。
- 每个种子颜色与边缘模板颜色一一对应。
Rule:
- 找到 bg 和灰色边界颜色；提取每个模板的默认图案（模板颜色＋3×3 与其上方 Yellow 标记）。
- 对每个不接触网格边界的连通区（以非灰色像素集合分割，包含 bg 和内部种子）找到唯一种子像素。
- 用种子相对区域几何中心确定方位（上/下/左/右），旋转对应模板。
- 从种子像素坐标作为周期原点，将旋转后 3×3 图案按模 3 平铺覆盖该连通区内的 bg 像素；模板 bg 位保持 bg，不改写灰色边界。
Key insight:
- 颜色决定模板样式，种子在区域中的方向决定模板朝向，二者必须同时推断才能唯一得到结果。"""

    arr = np.array(grid, dtype=np.int64)
    out = arr.copy()
    h, w = arr.shape

    if h == 0 or w == 0:
        return out

    bg = _infer_background(arr)
    wall = _infer_wall_color(arr, bg)
    walk_mask = arr != wall

    templates = _extract_templates(arr, bg)
    if not templates:
        return out

    for comp, touches_edge in _connected_components(walk_mask):
        if touches_edge or len(comp) < 1:
            continue

        # components should have one internal seed
        seeds = [tuple(p) for p in comp if arr[p[0], p[1]] != bg and arr[p[0], p[1]] != wall]
        if len(seeds) != 1:
            continue
        seed = seeds[0]
        seed_color = int(arr[seed[0], seed[1]])
        if seed_color not in templates:
            continue

        pattern = templates[seed_color]
        centers = comp.astype(np.float64).mean(axis=0)
        direction = _seed_direction(seed, centers[0], centers[1])
        rotated = _rotate_pattern(pattern, direction)

        sr, sc = seed
        for r, c in comp:
            if out[r, c] != bg:
                continue
            rr, cc = (r - sr) % 3, (c - sc) % 3
            if rotated[int(rr), int(cc)]:
                out[r, c] = seed_color

    return out


def _random_connected_set(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    max_cells = (r1 - r0 + 1) * (c1 - c0 + 1)
    if target <= 0 or target > max_cells:
        target = max_cells
    sr = int(rng.integers(r0, r1 + 1))
    sc = int(rng.integers(c0, c1 + 1))

    cells = [(sr, sc)]
    cell_set = {(sr, sc)}
    frontier = [(sr, sc)]
    attempts = 0
    while len(cell_set) < target and frontier:
        attempts += 1
        if attempts > 5000:
            return None
        idx = int(rng.integers(0, len(frontier)))
        r, c = frontier[idx]
        neigh = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cell_set:
                neigh.append((nr, nc))
        if not neigh:
            frontier.pop(idx)
            continue
        nr, nc = neigh[int(rng.integers(0, len(neigh)))]
        cell_set.add((nr, nc))
        frontier.append((nr, nc))
        cells.append((nr, nc))

    if len(cell_set) < target:
        return None
    return np.array(sorted(cell_set), dtype=np.int16)


def _wall_cells(cells: np.ndarray, h: int, w: int) -> np.ndarray:
    cell_set = set((int(r), int(c)) for r, c in cells)
    walls = set()
    for r, c in cell_set:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) not in cell_set:
                walls.add((nr, nc))
    return np.array(sorted(walls), dtype=np.int16)


def _choose_seed(cells: np.ndarray, direction: str, rng: np.random.Generator) -> Tuple[int, int]:
    rows = cells[:, 0].astype(float)
    cols = cells[:, 1].astype(float)
    cr, cc = rows.mean(), cols.mean()

    if direction == "up":
        idxs = np.where(rows < cr)[0]
    elif direction == "down":
        idxs = np.where(rows > cr)[0]
    elif direction == "left":
        idxs = np.where(cols < cc)[0]
    else:
        idxs = np.where(cols > cc)[0]

    if len(idxs) == 0:
        idxs = np.arange(len(cells))
    choice = int(rng.choice(idxs))
    return int(cells[choice, 0]), int(cells[choice, 1])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = 0
    grey = 1
    yellow = 4
    template_colors = [2, 3, 5, 6, 7, 8, 9]

    patterns = [
        np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
    ]

    for _ in range(200):
        H = int(rng.integers(20, 31))
        W = int(rng.integers(20, 31))
        grid = np.full((H, W), bg, dtype=np.int64)

        num_regions = int(rng.integers(1, 5))
        num_regions = min(num_regions, len(template_colors))
        chosen_colors = template_colors.copy()
        rng.shuffle(chosen_colors)
        chosen_colors = chosen_colors[:num_regions]

        occupied = np.zeros((H, W), dtype=bool)

        # place top-edge templates with yellow marker above each template
        marker_row = 0
        for color in chosen_colors:
            pattern = patterns[int(rng.integers(0, len(patterns)))]
            placed = False
            for _ in range(200):
                marker_col = int(rng.integers(1, W - 1))
                tc = marker_col - 1
                if tc < 0 or tc + 3 > W:
                    continue
                if occupied[marker_row, marker_col]:
                    continue
                if occupied[1:4, tc : tc + 3].any():
                    continue

                grid[marker_row, marker_col] = yellow
                for rr in range(3):
                    for cc in range(3):
                        if pattern[rr, cc]:
                            grid[1 + rr, tc + cc] = color
                occupied[marker_row, marker_col] = True
                occupied[1:4, tc : tc + 3] = True
                placed = True
                break
            if not placed:
                break
        else:
            # all templates placed successfully
            # block non-bg cells to avoid overlaps
            occupied |= (grid != bg)

            success_count = 0
            for i, color in enumerate(chosen_colors):
                placed_region = False
                for _ in range(800):
                    ih = int(rng.integers(4, 11))
                    iw = int(rng.integers(4, 11))
                    target = int(rng.integers(16, ih * iw + 1))

                    r0_min = 5
                    r1_max = H - ih - 2
                    c0_min = 1
                    c1_max = W - iw - 1
                    if r0_min > r1_max or c0_min >= c1_max:
                        continue
                    r0 = int(rng.integers(r0_min, r1_max + 1))
                    c0 = int(rng.integers(c0_min, c1_max + 1))
                    r1 = r0 + ih - 1
                    c1 = c0 + iw - 1

                    cells = _random_connected_set(r0, c0, r1, c1, target, rng)
                    if cells is None or len(cells) < 16:
                        continue

                    walls = _wall_cells(cells, H, W)

                    if walls.size == 0:
                        continue

                    inter = False
                    if occupied[cells[:, 0], cells[:, 1]].any() or occupied[walls[:, 0], walls[:, 1]].any():
                        inter = True
                    if inter:
                        continue

                    # ensure we get a closed region (walls and interior not on border)
                    if (
                        cells[:, 0].min() <= 0
                        or cells[:, 0].max() >= H - 1
                        or cells[:, 1].min() <= 0
                        or cells[:, 1].max() >= W - 1
                    ):
                        continue

                    if i < len(chosen_colors):
                        direction = rng.choice(["up", "down", "left", "right"])
                        sr, sc = _choose_seed(cells, direction, rng)
                        if sr is None:
                            continue
                        grid[walls[:, 0], walls[:, 1]] = grey
                        grid[sr, sc] = color
                        occupied[cells[:, 0], cells[:, 1]] = True
                        occupied[walls[:, 0], walls[:, 1]] = True
                        placed_region = True
                        success_count += 1
                        break
                if not placed_region:
                    break

            if success_count == num_regions:
                return grid

    raise RuntimeError("Failed to generate valid puzzle instance")

