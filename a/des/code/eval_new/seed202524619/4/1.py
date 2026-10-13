"""Scale and transfer seed frame patterns to matching blank frames by border color.
Input structure:
- 黑色是背景。
- 网格中有多个单色、1 格边框厚度的空心矩形框架。
- 其中一部分框架内部包含若干非黑种子图案像素；另外对应的空白框架内部全为黑色。
- 一个空白框架的边框颜色与其对应种子框架边框颜色一致，并且二者内框尺寸成整数倍关系。
Rule:
- 识别所有单色空心矩形边框，按边框颜色建立分组。
- 对每种颜色，在“有种子图案”的框架与“空白”框架之间一一配对。
- 令 k = 空白框架内高宽与种子框架内高宽的比例（两者必须一致）。
- 使用最近邻方式把种子内区块每个像素放大到 k×k 方块，并将结果填充到对应空白框架内区块。
- 其他区域保持不变，原始种子框架与其它像素不改动。
Key insight:
- 只有颜色配对与几何比例同时满足，才能决定“来自哪一个种子框架”，两层推理缺一不可。"""

import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

BLACK = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _extract_frames(grid: np.ndarray) -> List[Dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    frames: List[Dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color == BLACK or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells: List[Tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or grid[nr, nc] != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            rows = [x for x, _ in cells]
            cols = [y for _, y in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            frame_h = r1 - r0 + 1
            frame_w = c1 - c0 + 1
            if frame_h < 3 or frame_w < 3:
                continue

            # Validate strict 1-cell-thick hollow rectangle border.
            if len(cells) != 2 * (frame_h + frame_w - 2):
                continue
            for rr, cc in cells:
                if rr == r0 or rr == r1 or cc == c0 or cc == c1:
                    continue
                # interior pixel cannot be border color.
                break
            else:
                # Check full perimeter is border color.
                ok = True
                for rr in range(r0, r1 + 1):
                    if grid[rr, c0] != color or grid[rr, c1] != color:
                        ok = False
                        break
                if ok:
                    for cc in range(c0, c1 + 1):
                        if grid[r0, cc] != color or grid[r1, cc] != color:
                            ok = False
                            break
                if not ok:
                    continue

                inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                frames.append(
                    {
                        "color": int(color),
                        "r0": r0,
                        "r1": r1,
                        "c0": c0,
                        "c1": c1,
                        "inner": inner,
                        "has_seed": np.any(inner != BLACK),
                    }
                )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    inp = np.asarray(grid)
    if inp.ndim != 2:
        return inp.copy()

    out = inp.copy()
    frames = _extract_frames(inp)

    groups: Dict[int, Dict[str, List[Dict[str, object]]]] = {}
    for f in frames:
        color = f["color"]
        if color not in groups:
            groups[color] = {"seed": [], "blank": []}
        key = "seed" if f["has_seed"] else "blank"
        groups[color][key].append(f)

    for color, bucket in groups.items():
        seed_frames = bucket["seed"]
        blank_frames = bucket["blank"]
        pair_count = min(len(seed_frames), len(blank_frames))

        for i in range(pair_count):
            seed = seed_frames[i]
            blank = blank_frames[i]

            seed_inner = seed["inner"]
            ih, iw = seed_inner.shape
            bh = blank["r1"] - blank["r0"] - 1
            bw = blank["c1"] - blank["c0"] - 1

            if ih <= 0 or iw <= 0 or bh <= 0 or bw <= 0:
                continue
            if bh % ih != 0 or bw % iw != 0:
                continue

            k_h = bh // ih
            k_w = bw // iw
            if k_h <= 0 or k_w <= 0 or k_h != k_w:
                continue
            k = k_h

            scaled = np.repeat(np.repeat(seed_inner, k, axis=0), k, axis=1)
            if scaled.shape != (bh, bw):
                continue
            out[blank["r0"] + 1 : blank["r1"], blank["c0"] + 1 : blank["c1"]] = scaled

    return out


def _can_place(
    occ: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    margin: int = 1,
) -> bool:
    H, W = occ.shape
    r0 = max(0, top - margin)
    r1 = min(H, top + h + margin)
    c0 = max(0, left - margin)
    c1 = min(W, left + w + margin)
    return not occ[r0:r1, c0:c1].any()


def _draw_border(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    r1 = top + h - 1
    c1 = left + w - 1
    grid[top, left : c1 + 1] = color
    grid[r1, left : c1 + 1] = color
    grid[top : r1 + 1, left] = color
    grid[top : r1 + 1, c1] = color


def _random_seed_pattern(h: int, w: int, colors: List[int]) -> np.ndarray:
    pattern = np.zeros((h, w), dtype=int)
    branch = random.choice(("sparse", "blocks", "cross"))

    if branch == "sparse":
        p = random.uniform(0.2, 0.6)
        for i in range(h):
            for j in range(w):
                if random.random() < p:
                    pattern[i, j] = random.choice(colors)

    elif branch == "blocks":
        block_count = random.randint(1, 3)
        for _ in range(block_count):
            r0 = random.randint(0, h - 1)
            r1 = random.randint(r0, h - 1)
            c0 = random.randint(0, w - 1)
            c1 = random.randint(c0, w - 1)
            pattern[r0 : r1 + 1, c0 : c1 + 1] = random.choice(colors)

    else:
        row = random.randint(0, h - 1)
        col = random.randint(0, w - 1)
        pattern[row, :] = random.choice(colors)
        pattern[:, col] = random.choice(colors)
        if h >= 2 and w >= 2 and random.random() < 0.75:
            rr = random.randint(0, h - 1)
            cc = random.randint(0, w - 1)
            pattern[rr, cc] = random.choice(colors)

    if not np.any(pattern != 0):
        pattern[random.randint(0, h - 1), random.randint(0, w - 1)] = colors[0]

    return pattern


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(14, 30)
        w = random.randint(14, 30)
        grid = np.zeros((h, w), dtype=int)
        occupancy = np.zeros_like(grid, dtype=bool)

        pair_count = random.choice((1, 2, 3))
        pair_colors = random.sample(PALETTE, pair_count)

        pair_specs = []
        possible_scales = (2, 3)
        for color in pair_colors:
            scale = random.choice(possible_scales)
            seed_h = random.randint(2, 4)
            seed_w = random.randint(2, 4)
            blank_h = seed_h * scale + 2
            blank_w = seed_w * scale + 2
            seed_h_rect = seed_h + 2
            seed_w_rect = seed_w + 2
            pair_specs.append(
                {
                    "color": color,
                    "scale": scale,
                    "seed_h": seed_h,
                    "seed_w": seed_w,
                    "seed_rect": (seed_h_rect, seed_w_rect),
                    "blank_rect": (blank_h, blank_w),
                }
            )

        # Randomized placement order to increase layout variation.
        placements: List[Tuple[int, str, int, int, int, int]] = []
        items = []
        for idx, ps in enumerate(pair_specs):
            sh, sw = ps["seed_rect"]
            bh, bw = ps["blank_rect"]
            items.append((idx, "seed", sh, sw))
            items.append((idx, "blank", bh, bw))
        random.shuffle(items)

        pos = {}
        failed = False
        for idx, kind, ph, pw in items:
            placed = False
            for _ in range(400):
                top = random.randint(0, max(0, h - ph))
                left = random.randint(0, max(0, w - pw))
                if _can_place(occupancy, top, left, ph, pw, margin=1):
                    occupancy[top : top + ph, left : left + pw] = True
                    pos[(idx, kind)] = (top, left)
                    placements.append((idx, kind, top, left, ph, pw))
                    placed = True
                    break
            if not placed:
                failed = True
                break

        if failed:
            continue

        # Keep all specified constraints satisfied by construction.
        for idx, ps in enumerate(pair_specs):
            color = ps["color"]
            seed_top, seed_left = pos[(idx, "seed")]
            blank_top, blank_left = pos[(idx, "blank")]
            seed_h, seed_w = ps["seed_rect"]
            blank_h, blank_w = ps["blank_rect"]
            seed_pattern_h = ps["seed_h"]
            seed_pattern_w = ps["seed_w"]

            pattern_colors = [c for c in PALETTE if c != color]
            if len(pattern_colors) < 1:
                pattern_colors = [1]
            pattern = _random_seed_pattern(seed_pattern_h, seed_pattern_w, pattern_colors)

            _draw_border(grid, seed_top, seed_left, seed_h, seed_w, color)
            _draw_border(grid, blank_top, blank_left, blank_h, blank_w, color)
            r1 = seed_top + seed_h - 1
            c1 = seed_left + seed_w - 1
            grid[seed_top + 1 : r1, seed_left + 1 : c1] = pattern
            # blank interior remains black

        return grid

    # Fallback to avoid pathological placement failure while keeping solvability.
    fallback_h = 20
    fallback_w = 20
    fallback_grid = np.zeros((fallback_h, fallback_w), dtype=int)
    color = PALETTE[0]
    _draw_border(fallback_grid, 1, 1, 3 + 2, 3 + 2, color)
    pattern = np.array(
        [
            [2, 0, 2],
            [0, 2, 0],
            [2, 2, 2],
        ],
        dtype=int,
    )
    fallback_grid[2:5, 2:5] = pattern
    _draw_border(fallback_grid, 12, 2, (3 * 3) + 2, (3 * 3) + 2, color)
    return fallback_grid


if __name__ == "__main__":
    for _ in range(3):
        inp = generate_input()
        out = main(inp)
        print(inp.shape, inp.sum(), out.sum())
