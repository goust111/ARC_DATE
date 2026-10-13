"""Fill framed regions by deriving each region color from seed statistics, then leak a ray from its gap.

Input structure:
- 背景为 black(0)。
- 每个区域由单像素灰色(默认5)矩形边框围成，边框恰有一个缺口像素。
- 区域内部主要是 black，夹杂若干不同颜色的单像素种子点，种子位于边框内而边框上。

Rule:
- 对每个灰框区域，统计内部种子颜色的频次。颜色种类数记为 K，出现次数恰为 K 的颜色为候选填充色；若有多个候选，选择离缺口位置最近的那个颜色。
- 将该区域内部所有 black 像素改为该填充色（种子颜色保持不变）。
- 从缺口处沿缺口外侧方向发射宽 1 像素射线，射线颜色为填充色，并一直延伸到边界外或再次遇到灰框像素；多个射线按区域从上到下、从左到右顺序绘制，后绘制覆盖先绘制。

Key insight:
- 仅凭缺口无法得到射线颜色，必须先通过“颜色种类数 K 与出现次数 K 的颜色”这一步反推每个区域的代表色，再依据缺口方向画射线。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

import numpy as np

BG_COLOR = 0
DEFAULT_FRAME_COLOR = 5
SEED_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _frame_candidates(grid: np.ndarray, frame_color: int) -> List[Tuple[int, int, int, int, Tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == frame_color
    visited = np.zeros_like(mask, dtype=bool)
    candidates: List[Tuple[int, int, int, int, Tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue

            q = deque([(sr, sc)])
            visited[sr, sc] = True
            coords = [(sr, sc)]

            while q:
                r, c = q.popleft()
                if r > 0 and mask[r - 1, c] and not visited[r - 1, c]:
                    visited[r - 1, c] = True
                    q.append((r - 1, c))
                    coords.append((r - 1, c))
                if r + 1 < h and mask[r + 1, c] and not visited[r + 1, c]:
                    visited[r + 1, c] = True
                    q.append((r + 1, c))
                    coords.append((r + 1, c))
                if c > 0 and mask[r, c - 1] and not visited[r, c - 1]:
                    visited[r, c - 1] = True
                    q.append((r, c - 1))
                    coords.append((r, c - 1))
                if c + 1 < w and mask[r, c + 1] and not visited[r, c + 1]:
                    visited[r, c + 1] = True
                    q.append((r, c + 1))
                    coords.append((r, c + 1))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue

            expected: set[tuple[int, int]] = set()
            for c in range(c0, c1 + 1):
                expected.add((r0, c))
                expected.add((r1, c))
            for r in range(r0 + 1, r1):
                expected.add((r, c0))
                expected.add((r, c1))

            coord_set = set(coords)
            if len(coord_set - expected) > 0:
                continue

            missing = expected - coord_set
            if len(missing) != 1:
                continue

            (gr, gc) = next(iter(missing))
            if (gr in (r0, r1) and gc in (c0, c1)):
                continue

            candidates.append((r0, c0, r1, c1, (gr, gc)))

    return candidates


def _select_fill_color(grid: np.ndarray, region: Tuple[int, int, int, int], opening: Tuple[int, int], frame_color: int) -> Optional[int]:
    r0, c0, r1, c1 = region
    seed_positions: Dict[int, List[Tuple[int, int]]] = {}

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(grid[r, c])
            if v == BG_COLOR or v == frame_color:
                continue
            seed_positions.setdefault(v, []).append((r, c))

    if not seed_positions:
        return None

    k = len(seed_positions)
    candidates = [color for color, pos in seed_positions.items() if len(pos) == k]
    if candidates:
        if len(candidates) == 1:
            return candidates[0]

        gr, gc = opening
        best = candidates[0]
        best_dist = abs(seed_positions[best][0][0] - gr) + abs(seed_positions[best][0][1] - gc)
        for color in candidates[1:]:
            dist = min(abs(r - gr) + abs(c - gc) for r, c in seed_positions[color])
            if dist < best_dist or (dist == best_dist and color < best):
                best = color
                best_dist = dist
        return best

    # Fallback: if no count==K color exists, keep the most frequent internal color.
    best = next(iter(seed_positions.items()))
    for color, pos in seed_positions.items():
        if len(pos) > len(best[1]) or (len(pos) == len(best[1]) and color < best[0]):
            best = (color, pos)
    return best[0]


def _ray_direction(r0: int, c0: int, r1: int, c1: int, opening: Tuple[int, int]) -> Tuple[int, int]:
    gr, gc = opening
    if gr == r0:
        return -1, 0
    if gr == r1:
        return 1, 0
    if gc == c0:
        return 0, -1
    return 0, 1


def main(grid: np.ndarray) -> np.ndarray:
    grid_np = np.asarray(grid)
    out = grid_np.copy()
    if grid_np.ndim != 2:
        return out

    colors, counts = np.unique(grid_np, return_counts=True)
    non_bg = colors != BG_COLOR
    if not np.any(non_bg):
        return out
    frame_color = int(colors[non_bg][int(np.argmax(counts[non_bg]))])

    regions = _frame_candidates(out, frame_color)
    regions.sort(key=lambda item: (item[0], item[1]))  # top to bottom, left to right

    for r0, c0, r1, c1, opening in regions:
        fill_color = _select_fill_color(out, (r0, c0, r1, c1), opening, frame_color)
        if fill_color is None:
            continue

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == BG_COLOR:
                    out[r, c] = fill_color

        dr, dc = _ray_direction(r0, c0, r1, c1, opening)
        rr, cc = opening
        h, w = out.shape
        while 0 <= rr < h and 0 <= cc < w:
            if out[rr, cc] == frame_color:
                break
            out[rr, cc] = fill_color
            rr += dr
            cc += dc

    return out


def _set_if_available(
    arr: np.ndarray,
    positions: Iterable[Tuple[int, int]],
    value: int,
    occupied: np.ndarray,
    blocked: Optional[np.ndarray] = None,
) -> None:
    h, w = arr.shape
    for r, c in positions:
        if 0 <= r < h and 0 <= c < w:
            arr[r, c] = value
            if blocked is not None:
                blocked[r, c] = True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    target_frame = DEFAULT_FRAME_COLOR
    for _ in range(80):
        h = int(rng.integers(14, 30))
        w = int(rng.integers(14, 30))
        grid = np.full((h, w), BG_COLOR, dtype=np.int64)
        blocked = np.zeros_like(grid, dtype=bool)

        target_regions = int(rng.integers(1, 5))
        placed = 0

        for _ in range(target_regions):
            for _ in range(120):
                ih = int(rng.integers(2, 8))
                iw = int(rng.integers(2, 8))
                fh = ih + 2
                fw = iw + 2
                if fh > h - 1 or fw > w - 1:
                    continue

                r0 = int(rng.integers(1, h - fh))
                c0 = int(rng.integers(1, w - fw))
                r1 = r0 + fh - 1
                c1 = c0 + fw - 1

                r_min = max(0, r0 - 1)
                r_max = min(h, r1 + 2)
                c_min = max(0, c0 - 1)
                c_max = min(w, c1 + 2)
                if blocked[r_min:r_max, c_min:c_max].any():
                    continue

                open_dir = int(rng.integers(4))
                if open_dir == 0:  # top
                    gr = r0
                    gc = int(rng.integers(c0 + 1, c1))
                elif open_dir == 1:  # bottom
                    gr = r1
                    gc = int(rng.integers(c0 + 1, c1))
                elif open_dir == 2:  # left
                    gr = int(rng.integers(r0 + 1, r1))
                    gc = c0
                else:  # right
                    gr = int(rng.integers(r0 + 1, r1))
                    gc = c1

                # Draw rectangular frame with one missing pixel.
                for c in range(c0, c1 + 1):
                    if not (r0 == gr and c == gc):
                        grid[r0, c] = target_frame
                    if not (r1 == gr and c == gc):
                        grid[r1, c] = target_frame
                for r in range(r0 + 1, r1):
                    if not (r == gr and c0 == gc):
                        grid[r, c0] = target_frame
                    if not (r == gr and c1 == gc):
                        grid[r, c1] = target_frame

                interior: List[Tuple[int, int]] = [
                    (r, c)
                    for r in range(r0 + 1, r1)
                    for c in range(c0 + 1, c1)
                ]

                if len(interior) < 4:
                    continue

                interior_size = len(interior)
                max_k = min(3, interior_size // 2)
                if max_k < 1:
                    continue

                for _ in range(40):
                    k = int(rng.integers(1, max_k + 1))
                    colors = rng.choice(SEED_PALETTE, size=k, replace=False)

                    tie = k > 1 and rng.random() < 0.5
                    counts: Dict[int, int] = {}
                    if tie:
                        cA, cB = int(colors[0]), int(colors[1])
                        counts[cA] = k
                        counts[cB] = k
                        for c in colors[2:]:
                            cand = 1
                            while True:
                                cand = int(rng.integers(1, 4))
                                if cand != k:
                                    break
                            counts[int(c)] = cand
                    else:
                        counts[int(colors[0])] = k
                        for c in colors[1:]:
                            cand = int(rng.integers(1, 4))
                            if cand == k:
                                cand = 3 if k != 3 else 2
                            counts[int(c)] = cand

                    total = sum(counts.values())
                    if total <= interior_size:
                        break

                else:
                    continue

                rng.shuffle(interior)
                if sum(counts.values()) > len(interior):
                    continue

                # Put tie colors with controlled near/far order to trigger deterministic distance-based tie-break.
                remaining = interior.copy()
                if tie and k > 1:
                    by_dist = sorted(remaining, key=lambda p: abs(p[0] - gr) + abs(p[1] - gc))
                    near = by_dist[0]
                    far = by_dist[-1]
                    remaining = [p for p in remaining if p != near and p != far]
                    # place cA and cB with the closest/ farthest constraints
                    ordered_colors = [int(colors[0]), int(colors[1])] + [int(c) for c in colors[2:]]
                else:
                    ordered_colors = [int(c) for c in colors]

                for color in ordered_colors:
                    need = counts[color]
                    forced = None

                    if tie and k > 1 and color == int(colors[0]) and need > 0:
                        forced = near
                        need -= 1
                    if tie and k > 1 and color == int(colors[1]) and need > 0:
                        forced = far
                        need -= 1

                    if need > len(remaining):
                        break

                    picks = []
                    if forced is not None:
                        picks.append(forced)
                    if need > 0:
                        idx = rng.choice(len(remaining), size=need, replace=False)
                        if np.isscalar(idx):
                            idx = [int(idx)]
                        else:
                            idx = idx.tolist()
                        for i in sorted(idx, reverse=True):
                            picks.append(remaining[i])
                            del remaining[i]

                    for rr, cc in picks:
                        grid[rr, cc] = color
                else:
                    pass

                blocked[r0:r1 + 1, c0:c1 + 1] = True
                placed += 1
                break

        if placed >= 1:
            return grid

    # Fallback: build one small valid instance to avoid pathological generation failure.
    grid = np.zeros((18, 18), dtype=np.int64)
    r0, c0 = 5, 5
    ih, iw = 4, 5
    r1, c1 = r0 + ih + 1, c0 + iw + 1
    for c in range(c0, c1 + 1):
        grid[r0, c] = target_frame
        grid[r1, c] = target_frame
    for r in range(r0 + 1, r1):
        grid[r, c0] = target_frame
        grid[r, c1] = target_frame
    grid[r0, c0 + 2] = BG_COLOR
    grid[r0 + 1, c0 + 2] = 1
    grid[r0 + 2, c0 + 3] = 2
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
