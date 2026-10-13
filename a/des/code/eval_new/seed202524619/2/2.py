"""将钥匙按同色容器与缺口方向完成方向匹配后嵌入容器。

Input structure:
背景色为网格中出现次数最多的颜色，另外有若干种非背景颜色。
每种非背景颜色恰好对应一个对象：一个大空心方形边框容器（边框厚度为1且某一侧缺口宽2）和一个钥匙图形对象。
钥匙图形与其容器同色，颜色块通常为 5 像素的 L/T 风格形状，并且有可辨识朝向。

Rule:
对每种颜色先分离出容器与钥匙。
根据容器边框缺失的两格位置确定缺口方向。
识别钥匙的当前朝向；将钥匙在 90° 倍数内旋转到与缺口方向一致后平移。
平移时使钥匙的突出臂（两格宽）精确覆盖缺口位置，钥匙其余部分位于容器内部。
保留容器边框，并将钥匙从原位置移除后放入目标容器。

Key insight:
按颜色建立一一配对后，关键在于从钥匙几何形状恢复方向，再按缺口方向进行旋转对齐，之后做受约束的刚体平移。
"""

import numpy as np
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# 5-cell key templates, normalized coordinates (min row/col = 0).
# Each template has a two-cell arm on the corresponding direction side.
KEY_TEMPLATES: Dict[int, Tuple[Tuple[Tuple[int, int], ...], ...]] = {
    UP: ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    RIGHT: ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
    DOWN: ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    LEFT: ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
}
KEY_TEMPLATE_SETS: Dict[int, Set[Tuple[int, int]]] = {
    d: set(v) for d, v in KEY_TEMPLATES.items()
}
KEY_ARM_CELLS: Dict[int, Set[Tuple[int, int]]] = {
    UP: {(0, 1), (0, 2)},
    RIGHT: {(1, 1), (2, 1)},
    DOWN: {(1, 0), (1, 1)},
    LEFT: {(0, 0), (1, 0)},
}


def _normalize_coords(coords: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    coords = list(coords)
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _connected_components(mask: np.ndarray) -> List[Dict[str, object]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: List[Tuple[int, int]] = []
            min_r = min_c = 10**9
            max_r = max_c = -1
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c
                nr = r - 1
                if nr >= 0 and mask[nr, c] and not visited[nr, c]:
                    visited[nr, c] = True
                    q.append((nr, c))
                nr = r + 1
                if nr < h and mask[nr, c] and not visited[nr, c]:
                    visited[nr, c] = True
                    q.append((nr, c))
                nc = c - 1
                if nc >= 0 and mask[r, nc] and not visited[r, nc]:
                    visited[r, nc] = True
                    q.append((r, nc))
                nc = c + 1
                if nc < w and mask[r, nc] and not visited[r, nc]:
                    visited[r, nc] = True
                    q.append((r, nc))

            comps.append(
                {
                    "coords": set(cells),
                    "bbox": (min_r, max_r, min_c, max_c),
                    "size": len(cells),
                }
            )
    return comps


def _frame_gap_info(comp: Dict[str, object]) -> Optional[Tuple[int, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    coords: Set[Tuple[int, int]] = comp["coords"]  # type: ignore[assignment]
    min_r, max_r, min_c, max_c = comp["bbox"]  # type: ignore[misc]
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    if h < 4 or w < 4:
        return None

    # Frame must lie on the rectangle border.
    for r, c in coords:
        if r != min_r and r != max_r and c != min_c and c != max_c:
            return None

    expected = 2 * (h + w) - 4
    if len(coords) != expected - 2:
        return None

    top_missing = [c for c in range(min_c, max_c + 1) if (min_r, c) not in coords]
    bottom_missing = [c for c in range(min_c, max_c + 1) if (max_r, c) not in coords]
    left_missing = [r for r in range(min_r, max_r + 1) if (r, min_c) not in coords]
    right_missing = [r for r in range(min_r, max_r + 1) if (r, max_c) not in coords]

    candidates: List[Tuple[int, Tuple[Tuple[int, int], Tuple[int, int]]]] = []

    if (
        len(top_missing) == 2
        and top_missing[0] + 1 == top_missing[1]
        and top_missing[0] >= min_c + 1
        and top_missing[1] <= max_c - 1
    ):
        c0 = top_missing[0]
        candidates.append((UP, ((min_r, c0), (min_r, c0 + 1))))

    if (
        len(bottom_missing) == 2
        and bottom_missing[0] + 1 == bottom_missing[1]
        and bottom_missing[0] >= min_c + 1
        and bottom_missing[1] <= max_c - 1
    ):
        c0 = bottom_missing[0]
        candidates.append((DOWN, ((max_r, c0), (max_r, c0 + 1))))

    if (
        len(left_missing) == 2
        and left_missing[0] + 1 == left_missing[1]
        and left_missing[0] >= min_r + 1
        and left_missing[1] <= max_r - 1
    ):
        r0 = left_missing[0]
        candidates.append((LEFT, ((r0, min_c), (r0 + 1, min_c))))

    if (
        len(right_missing) == 2
        and right_missing[0] + 1 == right_missing[1]
        and right_missing[0] >= min_r + 1
        and right_missing[1] <= max_r - 1
    ):
        r0 = right_missing[0]
        candidates.append((RIGHT, ((r0, max_c), (r0 + 1, max_c))))

    if len(candidates) != 1:
        return None

    return candidates[0]


def _detect_key_dir(comp: Dict[str, object]) -> Optional[int]:
    coords = comp["coords"]  # type: ignore[assignment]
    norm = _normalize_coords(coords)
    for d, tpl in KEY_TEMPLATE_SETS.items():
        if norm == tpl:
            return d
    return None


def _translate_set(coords: Set[Tuple[int, int]], dr: int, dc: int) -> Set[Tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _fits_in_frame(
    shape: Set[Tuple[int, int]],
    frame: Dict[str, object],
    arm_target: Set[Tuple[int, int]],
    gap: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[Set[Tuple[int, int]]]:
    min_r, max_r, min_c, max_c = frame["bbox"]  # type: ignore[misc]
    gap_cells = set(gap)
    arm_sorted = sorted(arm_target)
    gap_sorted = sorted(gap_cells)

    if len(arm_sorted) != 2 or len(gap_sorted) != 2:
        return None

    dr_candidates = {gap_sorted[i][0] - arm_sorted[i][0] for i in range(2)}
    dc_candidates = {gap_sorted[i][1] - arm_sorted[i][1] for i in range(2)}
    if len(dr_candidates) != 1 or len(dc_candidates) != 1:
        return None

    dr = next(iter(dr_candidates))
    dc = next(iter(dc_candidates))
    translated = _translate_set(shape, dr, dc)
    translated_arm = _translate_set(arm_target, dr, dc)

    if translated_arm != gap_cells:
        return None

    for r, c in translated:
        if not (min_r <= r <= max_r and min_c <= c <= max_c):
            return None
    return translated


def main(grid: np.ndarray) -> np.ndarray:
    """Solve ARC transformation for keyed shape insertion."""
    g = np.array(grid, dtype=int)
    out = g.copy()

    if g.ndim != 2 or g.size == 0:
        return out

    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    colors = [int(c) for c in vals if int(c) != bg]

    for color in colors:
        mask = g == color
        comps = _connected_components(mask)
        if len(comps) != 2:
            continue

        frame = None
        key = None
        for comp in comps:
            f = _frame_gap_info(comp)
            if f is not None:
                frame = comp
                break

        if frame is None:
            frame = max(comps, key=lambda x: x["size"])
            key = min(comps, key=lambda x: x["size"])
        else:
            key = comps[0] if comps[0] is not frame else comps[1]

        frame_gap = _frame_gap_info(frame)
        if frame_gap is None:
            continue
        gap_dir, gap_cells = frame_gap

        key_dir = _detect_key_dir(key)
        if key_dir is None:
            continue

        target_shape = KEY_TEMPLATE_SETS[gap_dir]
        target_arm = KEY_ARM_CELLS[gap_dir]

        placed = _fits_in_frame(target_shape, frame, target_arm, gap_cells)
        if placed is None:
            continue

        # remove original key
        for r, c in key["coords"]:
            out[r, c] = bg

        # insert rotated/matched key
        for r, c in placed:
            out[r, c] = color

    return out


def _frame_cells(
    top: int,
    left: int,
    h: int,
    w: int,
    gap_dir: int,
    gap_start: int,
) -> Tuple[Set[Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]]]:
    cells: Set[Tuple[int, int]] = set()
    if gap_dir == UP:
        gap = ((top, gap_start), (top, gap_start + 1))
    elif gap_dir == DOWN:
        gap = ((top + h - 1, gap_start), (top + h - 1, gap_start + 1))
    elif gap_dir == LEFT:
        gap = ((gap_start, left), (gap_start + 1, left))
    else:
        gap = ((gap_start, left + w - 1), (gap_start + 1, left + w - 1))

    cells.update((top, c) for c in range(left, left + w))
    cells.update((top + h - 1, c) for c in range(left, left + w))
    cells.update((r, left) for r in range(top, top + h))
    cells.update((r, left + w - 1) for r in range(top, top + h))

    if gap_dir == UP:
        cells.remove((top, gap[0][1]))
        cells.remove((top, gap[1][1]))
    elif gap_dir == DOWN:
        cells.remove((top + h - 1, gap[0][1]))
        cells.remove((top + h - 1, gap[1][1]))
    elif gap_dir == LEFT:
        cells.remove((gap[0][0], left))
        cells.remove((gap[1][0], left))
    else:
        cells.remove((gap[0][0], left + w - 1))
        cells.remove((gap[1][0], left + w - 1))

    return cells, gap


def _place_pair(
    grid: np.ndarray,
    occ: np.ndarray,
    color: int,
    rng: np.random.Generator,
    require_diverse_dirs: bool,
    key_rotation_branch: int,
) -> bool:
    H, W = grid.shape
    for _ in range(250):
        h = int(rng.integers(7, 12))
        w = int(rng.integers(7, 12))

        if h >= H - 2 or w >= W - 2:
            continue

        top = int(rng.integers(1, H - h - 1))
        left = int(rng.integers(1, W - w - 1))
        gap_dir = int(rng.integers(0, 4))

        if gap_dir in (UP, DOWN):
            gap_start = int(rng.integers(left + 1, left + w - 2))
        else:
            gap_start = int(rng.integers(top + 1, top + h - 2))

        frame_cells, gap = _frame_cells(top, left, h, w, gap_dir, gap_start)

        if any(occ[r, c] for r, c in frame_cells):
            continue

        need_rotate = bool(rng.integers(0, 2) == 1)
        if key_rotation_branch == 0:
            key_dir = gap_dir
            need_rotate = False
        elif key_rotation_branch == 1:
            key_dir = rng.integers(0, 4)
            while key_dir == gap_dir:
                key_dir = (key_dir + 1) % 4
        else:
            key_dir = int(rng.integers(0, 4))
            if need_rotate and key_dir == gap_dir:
                key_dir = (key_dir + 1) % 4
            if (not need_rotate) and key_dir != gap_dir:
                need_rotate = True

        shape = KEY_TEMPLATE_SETS[key_dir]
        kH = max(r for r, _ in shape) + 1
        kW = max(c for _, c in shape) + 1

        # place key somewhere outside the frame in free space
        candidate_limit = 300
        # expand frame bounds by one cell to avoid accidental touching
        e_top = max(0, top - 1)
        e_left = max(0, left - 1)
        e_bottom = min(H - 1, top + h)
        e_right = min(W - 1, left + w)

        for _ in range(candidate_limit):
            kr = int(rng.integers(0, H - kH + 1))
            kc = int(rng.integers(0, W - kW + 1))

            key_cells = {(kr + r, kc + c) for r, c in shape}

            if any(occ[r, c] for r, c in key_cells):
                continue

            touches_frame = (
                not (kr + kH - 1 < e_top or kr > e_bottom or kc + kW - 1 < e_left or kc > e_right)
            )
            if touches_frame:
                continue

            for r, c in frame_cells:
                grid[r, c] = color
                occ[r, c] = True
            for r, c in key_cells:
                grid[r, c] = color
                occ[r, c] = True

            return True
    return False


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 5, 6]

    for _ in range(80):
        rng = np.random.default_rng()
        num_pairs = int(rng.integers(1, 3))
        H = int(rng.integers(18, 29))
        W = int(rng.integers(18, 29))
        if H > 30:
            H = 30
        if W > 30:
            W = 30

        grid = np.zeros((H, W), dtype=int)
        occ = np.zeros((H, W), dtype=bool)

        selected_colors = palette.copy()
        selected_colors = selected_colors[:num_pairs]
        rng.shuffle(selected_colors)

        branch_flag = int(rng.integers(0, 3))

        ok = True
        for color in selected_colors:
            if not _place_pair(
                grid,
                occ,
                color,
                rng,
                require_diverse_dirs=True,
                key_rotation_branch=branch_flag,
            ):
                ok = False
                break

        if not ok:
            continue

        # ensure all colors have exactly one frame+one key in this sample
        for color in selected_colors:
            comps = _connected_components(grid == color)
            if len(comps) != 2:
                ok = False
                break
        if not ok:
            continue

        return grid

    # deterministic fallback
    grid = np.zeros((22, 22), dtype=int)
    _place_pair(
        grid,
        np.zeros_like(grid, dtype=bool),
        1,
        np.random.default_rng(0),
        require_diverse_dirs=True,
        key_rotation_branch=2,
    )
    return grid

