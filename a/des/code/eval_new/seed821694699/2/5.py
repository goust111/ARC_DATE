from __future__ import annotations

import numpy as np
from typing import List, Tuple

# ARC standard color constants used by the puzzle statement.
BG_COLOR = 8  # azure
SHADOW_COLOR = 5  # grey


def _contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Convert sorted 1D indices into inclusive contiguous run ranges."""
    if indices.size == 0:
        return []

    runs: List[Tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx_i = int(idx)
        if idx_i == prev + 1:
            prev = idx_i
        else:
            runs.append((start, prev))
            start = idx_i
            prev = idx_i
    runs.append((start, prev))
    return runs


def _infer_colors(grid: np.ndarray) -> tuple[int, int, list[int]]:
    """Infer background/shadow/object colors from the grid."""
    values, counts = np.unique(grid, return_counts=True)
    bg_color = int(values[np.argmax(counts)])

    color_count = {int(v): int(c) for v, c in zip(values, counts)}
    candidates = [c for c in color_count if c != bg_color]
    shadow_color = SHADOW_COLOR if SHADOW_COLOR in candidates else (max(candidates, key=color_count.get) if candidates else bg_color)
    object_colors = [c for c in candidates if c != shadow_color]
    return bg_color, shadow_color, object_colors


def _infer_base_span(xs: np.ndarray, ys: np.ndarray) -> tuple[int, int] | None:
    """Find the widest contiguous x-span on the object's bottom row."""
    if ys.size == 0:
        return None

    base_y = int(ys.max())
    base_xs = np.sort(xs[ys == base_y].astype(int))
    runs = _contiguous_runs(base_xs)
    if not runs:
        return None

    return max(runs, key=lambda r: r[1] - r[0] + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    1) 解析颜色：识别背景色、阴影色和每个独立物体颜色。
    2) 读取每个物体：统计面积 N，以及在最低一行上的底边宽度 W。
    3) 统一向下投射：每列尝试向下放置 N 个 grey 像素，遇到其它物体像素直接跳过但不结束该列计数。
    4) 输出复原网格：保留原有物体与背景不变，把重建后的阴影以 grey 写回。

    Parameters
    ----------
    grid: np.ndarray
        输入 ARC 网格（整数颜色编码）。

    Returns
    -------
    np.ndarray
        目标阴影补全后的网格，与输入尺寸一致。

    Strategy/复杂度
    -------------
    逐物体逐列逐行向下计数补齐阴影。复杂度约 O(K*W*H)，
    其中 K 为物体总数，W/H 为网格宽高。
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("grid must be a 2D ndarray")
    h, w = arr.shape

    bg_color, shadow_color, object_colors = _infer_colors(arr)
    if not object_colors or bg_color == shadow_color:
        return arr.copy()

    object_mask = np.isin(arr, object_colors)
    out = arr.copy()

    for obj_color in object_colors:
        ys, xs = np.where(arr == obj_color)
        if ys.size == 0:
            continue

        area = int(ys.size)
        base_y = int(ys.max())
        span = _infer_base_span(xs, ys)
        if span is None:
            continue
        x0, x1 = span

        for x in range(x0, x1 + 1):
            placed = 0
            y = base_y + 1
            while placed < area and y < h:
                if object_mask[y, x]:
                    y += 1
                    continue

                out[y, x] = shadow_color
                placed += 1
                y += 1

    return out


def _place_objects_randomly(rng: np.random.Generator, h: int, w: int, max_objects: int) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)

    palette = [0, 1, 2, 3, 4, 6, 7, 9]
    rng.shuffle(palette)
    target_objects = int(rng.integers(2, min(max_objects, len(palette)) + 1))
    objects: list[dict] = []

    for idx in range(target_objects):
        color = palette[idx]
        if color == BG_COLOR or color == SHADOW_COLOR:
            continue

        placed = False
        for _ in range(200):
            obj_w = int(rng.integers(2, 6))
            obj_h = int(rng.integers(2, 5))
            area = obj_w * obj_h

            # Try to keep enough room under the object for at least a useful shadow section.
            if obj_h + area + 2 > h:
                continue

            x_max = w - obj_w
            if x_max < 1:
                continue
            y_limit = h - (obj_h + area + 1)
            if y_limit <= 1:
                # Fallback: accept any top position that at least fits object body.
                y_limit = h - obj_h
                if y_limit <= 1:
                    continue

            x0 = int(rng.integers(0, x_max + 1))
            y0 = int(rng.integers(1, y_limit + 1))
            if occupancy[y0 : y0 + obj_h, x0 : x0 + obj_w].any():
                continue

            occupancy[y0 : y0 + obj_h, x0 : x0 + obj_w] = True
            grid[y0 : y0 + obj_h, x0 : x0 + obj_w] = color

            objects.append(
                {
                    "color": color,
                    "x0": x0,
                    "x1": x0 + obj_w - 1,
                    "base": y0 + obj_h - 1,
                    "area": area,
                }
            )
            placed = True
            break

        # No strict requirement on exact target count; stop when we cannot place more.
        if not placed:
            continue

    return grid, objects


def _draw_truncated_shadows(grid: np.ndarray, objects: list[dict]) -> None:
    h, w = grid.shape
    if not objects:
        return

    occupancy = np.isin(grid, [obj["color"] for obj in objects])

    for obj in objects:
        base = obj["base"]
        length = obj["area"]
        for x in range(obj["x0"], obj["x1"] + 1):
            y = base + 1
            remaining = length
            while remaining > 0 and y < h:
                if occupancy[y, x]:
                    break
                grid[y, x] = SHADOW_COLOR
                remaining -= 1
                y += 1


def generate_input() -> np.ndarray:
    """Generate a random ARC-like input that follows the same color convention and rules."""
    rng = np.random.default_rng()
    h = int(rng.integers(14, 27))
    w = int(rng.integers(14, 27))

    grid, objects = _place_objects_randomly(rng, h, w, max_objects=4)
    _draw_truncated_shadows(grid, objects)
    return grid


if __name__ == "__main__":
    # Lightweight self-check.
    sample = generate_input()
    output = main(sample)
    assert sample.shape == output.shape
    obj_mask = ~((sample == BG_COLOR) | (sample == SHADOW_COLOR))
    assert np.array_equal(output[obj_mask], sample[obj_mask])
    print("self-check: ok", sample.shape)
