from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []
            while queue:
                x, y = queue.popleft()
                comp.append((x, y))
                for nr, nc in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _component_size_one(mask: np.ndarray, start: Tuple[int, int]) -> bool:
    h, w = mask.shape
    r, c = start
    if not (0 <= r < h and 0 <= c < w and mask[r, c]):
        return False
    q = deque([(r, c)])
    mask[r, c] = False
    visited = False
    count = 0
    while q:
        x, y = q.popleft()
        count += 1
        if count > 1:
            return False
        for nr, nc in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                mask[nr, nc] = False
                q.append((nr, nc))
    return count == 1


def _pattern_bbox(mask: np.ndarray, comp: Iterable[Tuple[int, int]]) -> np.ndarray:
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each right-side container with the corresponding fragment texture based on color counts.

    Input structure:
    - The background is the dominant color.
    - A full-height vertical separator color splits the grid into a left field and a right field.
    - Left field contains small connected fragment components; same-color components share the same 2-4-cell shape.
    - Right field contains equal-width open-top U-shaped grey containers, each with one marker pixel at the bottom center.
    - Each marker color matches one fragment color from the left field.

    Rule:
    - For each marker color, count the number of fragment components with that color in the left field (N).
    - From the marker, identify that container's interior bounds by the grey bottom border row.
    - Fill N rows upward from the container bottom inside the container width.
    - Use the fragment's representative shape as a repeating texture: repeat horizontally across the row and
      repeat from the bottom when vertical pattern height is exceeded.
    - Keep all other pixels unchanged (including left fragments and separators).

    Key insight:
    - The fragment quantity controls fill height, and the fragment shape independently controls texture.
    """
    inp = np.array(grid, dtype=int)
    h, w = inp.shape
    out = np.array(inp, copy=True)

    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]

    separator_col = None
    marker_color = None
    for color in sorted(colors):
        if color == bg:
            continue
        full_cols = [c for c in range(w) if np.all(inp[:, c] == color)]
        if full_cols:
            separator_col = full_cols[0]
            marker_color = color
            break
    if separator_col is None:
        # Fallback: keep behavior deterministic even for unexpected palettes.
        sorted_colors = [int(c) for c in colors if c != bg]
        marker_color = sorted_colors[0] if sorted_colors else bg
        separator_col = w // 2

    left = inp[:, :separator_col]
    right = inp[:, separator_col + 1 :]

    marker_positions: Dict[int, Tuple[int, int]] = {}
    for color in np.unique(right):
        if color == bg or color == marker_color:
            continue
        mask = (inp == color)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        for r, c in zip(ys, xs):
            if c <= separator_col:
                continue
            temp = np.array(mask, copy=True)
            if _component_size_one(temp, (r, c)):
                marker_positions[color] = (r, c)
                break

    for color, marker in marker_positions.items():
        comps = _connected_components(left == color)
        if not comps:
            continue
        count = len(comps)

        pattern = _pattern_bbox(inp, comps[0])
        ph, pw = pattern.shape
        if ph == 0 or pw == 0:
            continue

        mr, mc = marker
        if mr + 1 >= h or inp[mr + 1, mc] != marker_color:
            continue
        b = mr + 1

        left_wall = mc
        while left_wall - 1 >= 0 and inp[b, left_wall - 1] == marker_color:
            left_wall -= 1
        right_wall = mc
        while right_wall + 1 < w and inp[b, right_wall + 1] == marker_color:
            right_wall += 1
        if right_wall <= left_wall + 1:
            continue

        x0 = left_wall + 1
        x1 = right_wall - 1
        width = x1 - x0 + 1
        for row_idx in range(count):
            yr = b - 1 - row_idx
            if yr < 0:
                break
            p_row = ph - 1 - (row_idx % ph)
            for dx in range(width):
                sx = dx % pw
                if pattern[p_row, sx]:
                    out[yr, x0 + dx] = color

    return out


def _random_shape(rng: np.random.Generator, size: int) -> List[Tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        cr, cc = rng.choice(list(cells))
        nr, nc = cr + rng.integers(-1, 2), cc + rng.integers(-1, 2)
        if abs(nr - cr) + abs(nc - cc) != 1:
            continue
        if (nr, nc) not in cells:
            cells.add((nr, nc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _shape_signature(cells: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _can_place(
    occ: np.ndarray,
    shape: List[Tuple[int, int]],
    top: int,
    left: int,
) -> bool:
    h, w = occ.shape
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if occ[rr, cc]:
            return False
        for nr in (rr - 1, rr, rr + 1):
            for nc in (cc - 1, cc, cc + 1):
                if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                    return False
    return True


def _paint_shape(grid: np.ndarray, occ: np.ndarray, top: int, left: int, shape: List[Tuple[int, int]], color: int) -> None:
    for dr, dc in shape:
        rr = top + dr
        cc = left + dc
        grid[rr, cc] = color
        occ[rr, cc] = True


def generate_input() -> np.ndarray:
    """
    Randomly synthesise ARC-like instances for the fill-with-texture rule.
    """
    rng = np.random.default_rng()
    bg = 0
    gray = 8
    color_pool = [1, 2, 3, 4, 5, 6, 7, 9]

    for _ in range(5000):
        num_colors = int(rng.integers(3, 5))
        colors = rng.choice(color_pool, size=num_colors, replace=False).tolist()

        color_shapes: Dict[int, List[Tuple[int, int]]] = {}
        color_counts: Dict[int, int] = {}
        used_shape_sigs: set = set()
        for color in colors:
            count = int(rng.integers(1, 7))
            shape_size = int(rng.integers(2, 5))
            for _ in range(25):
                shape = _random_shape(rng, shape_size)
                sig = _shape_signature(shape)
                if sig not in used_shape_sigs:
                    break
                shape = _random_shape(rng, shape_size)
            used_shape_sigs.add(sig)
            color_shapes[color] = shape
            color_counts[color] = count

        max_cnt = max(color_counts.values())
        interior_w = int(rng.integers(3, 7))
        gap = 1
        container_total = interior_w + 2
        right_width = num_colors * container_total + max(0, (num_colors - 1) * gap)
        sep_limit = 29 - right_width
        if sep_limit < 8:
            continue

        h = int(rng.integers(14, 31))
        sep_col = int(rng.integers(8, sep_limit + 1))
        w = sep_col + 1 + right_width
        if w > 30 or h > 30 or interior_w >= sep_col:
            continue

        grid = np.full((h, w), bg, dtype=int)
        grid[:, sep_col] = gray
        left_occ = np.zeros((h, sep_col), dtype=bool)
        container_bottom = h - 1
        marker_row = h - 2
        if marker_row < 0 or sep_col <= interior_w:
            continue

        placed_ok = True
        for color in colors:
            shape = color_shapes[color]
            count = color_counts[color]
            for _ in range(count):
                shape_h = max(r for r, _ in shape) + 1
                shape_w = max(c for _, c in shape) + 1
                y_hi = h - 7
                if y_hi < 2:
                    y_hi = 2
                x_hi = sep_col - shape_w
                if x_hi < 0:
                    placed_ok = False
                    break
                attempts = 0
                found = False
                while attempts < 400:
                    attempts += 1
                    top = int(rng.integers(0, max(1, y_hi - shape_h + 1)))
                    left = int(rng.integers(0, x_hi + 1))
                    if _can_place(left_occ, shape, top, left):
                        _paint_shape(grid[:, :sep_col], left_occ, top, left, shape, color)
                        found = True
                        break
                if not found:
                    placed_ok = False
                    break
            if not placed_ok:
                break
        if not placed_ok:
            continue

        # Draw equal-width U-like containers and markers.
        color_order = colors.copy()
        rng.shuffle(color_order)
        max_stack_height = min(marker_row - 2, max_cnt + 7)
        container_top_min = max(1, marker_row - max_stack_height)
        for i, color in enumerate(color_order):
            cx0 = sep_col + 1 + i * (container_total + gap)
            cx1 = cx0 + container_total - 1
            if cx1 >= w:
                placed_ok = False
                break
            container_top = int(rng.integers(container_top_min, marker_row))
            for yy in range(container_top, h):
                grid[yy, cx0] = gray
                grid[yy, cx1] = gray
            grid[container_bottom, cx0:cx1 + 1] = gray
            marker_x = cx0 + 1 + interior_w // 2
            grid[marker_row, marker_x] = color
        if not placed_ok:
            continue

        return grid

    raise RuntimeError("unable to sample a valid puzzle instance")
