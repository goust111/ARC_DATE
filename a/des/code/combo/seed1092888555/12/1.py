from __future__ import annotations

import random
from collections import deque

import numpy as np


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _are_bbox_neighbors(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    row_gap = _interval_gap(r0a, r1a, r0b, r1b)
    col_gap = _interval_gap(c0a, c1a, c0b, c1b)
    same_band_rows = _intervals_overlap(r0a, r1a, r0b, r1b)
    same_band_cols = _intervals_overlap(c0a, c1a, c0b, c1b)
    return (same_band_rows and col_gap <= 1) or (same_band_cols and row_gap <= 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each object with a center glyph chosen by its bbox-neighbor count.

    Objects are 4-connected monochrome foreground components on a black background.
    Two objects are neighbors when their bounding boxes are cardinally adjacent:
    their row spans overlap and their column gap is at most 1, or vice versa.

    Each object is cleared and redrawn at its bbox center:
    - 0 neighbors: 3x3 solid square
    - 1 neighbor:  5-pixel plus
    - 2+ neighbors: 5-pixel diamond
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    comps = _components(g)

    for i, comp in enumerate(comps):
        neighbors = 0
        for j, other in enumerate(comps):
            if i != j and _are_bbox_neighbors(comp["bbox"], other["bbox"]):
                neighbors += 1

        r0, c0, r1, c1 = comp["bbox"]
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        color = int(comp["color"])

        if neighbors == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc2 = cr + dr, cc + dc
                    if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                        out[rr, cc2] = color
        elif neighbors == 1:
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc2 = cr + dr, cc + dc
                if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                    out[rr, cc2] = color
        else:
            for dr, dc in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc2 = cr + dr, cc + dc
                if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                    out[rr, cc2] = color

    return out


def _connected_masks_3x3() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area < 3 or area > 8:
            continue
        mask = np.array([(bits >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3)
        if not mask[1, 1]:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        seen = set()
        start = tuple(np.argwhere(mask)[0])
        q = deque([start])
        seen.add(start)
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3 and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        if len(seen) == area:
            masks.append(mask)
    return masks


_MASKS_3X3 = _connected_masks_3x3()
_CLUSTER_PATTERNS = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]


def _transform_pattern(pattern: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = list(pattern)
    h = 1 + max(r for r, _ in pts)
    w = 1 + max(c for _, c in pts)
    choice = random.randrange(8)
    out = []
    for r, c in pts:
        rr, cc = r, c
        if choice & 1:
            cc = w - 1 - cc
        if choice & 2:
            rr = h - 1 - rr
        if choice & 4:
            rr, cc = cc, rr
        out.append((rr, cc))
    r0 = min(r for r, _ in out)
    c0 = min(c for _, c in out)
    return {(r - r0, c - c0) for r, c in out}


def _coarse_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {}
    for cell in cells:
        count = 0
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cells:
                count += 1
        deg[cell] = count
    return deg


def _can_place_group(used: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in used:
            return False
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in used:
                return False
    return True


def _place_group_on_board(
    board_h: int, board_w: int, used: set[tuple[int, int]], pattern: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    pattern = _transform_pattern(pattern)
    ph = 1 + max(r for r, _ in pattern)
    pw = 1 + max(c for _, c in pattern)
    offsets = [(r0, c0) for r0 in range(board_h - ph + 1) for c0 in range(board_w - pw + 1)]
    random.shuffle(offsets)
    for r0, c0 in offsets:
        cells = {(r0 + r, c0 + c) for r, c in pattern}
        if _can_place_group(used, cells):
            return cells
    return None


def _draw_object(grid: np.ndarray, center_r: int, center_c: int, color: int, mask: np.ndarray) -> None:
    r0 = center_r - 1
    c0 = center_c - 1
    grid[r0 : r0 + 3, c0 : c0 + 3][mask] = int(color)


def _generate_candidate() -> np.ndarray | None:
    board_h = random.randint(4, 6)
    board_w = random.randint(4, 6)
    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    grid_h = margin_top * 2 + 4 * board_h - 1
    grid_w = margin_left * 2 + 4 * board_w - 1
    grid = np.zeros((grid_h, grid_w), dtype=int)

    group_patterns = [
        {(0, 0)},
        random.choice([{(0, 0), (0, 1)}, {(0, 0), (1, 0)}]),
        random.choice(_CLUSTER_PATTERNS),
    ]
    extra_count = random.randint(0, 2)
    extras = [{(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 0)}] + _CLUSTER_PATTERNS
    for _ in range(extra_count):
        group_patterns.append(random.choice(extras))

    used: set[tuple[int, int]] = set()
    all_cells: set[tuple[int, int]] = set()
    for pattern in group_patterns:
        placed = _place_group_on_board(board_h, board_w, used, pattern)
        if placed is None:
            return None
        used |= placed
        all_cells |= placed

    object_cells = sorted(all_cells)
    if not (4 <= len(object_cells) <= 8):
        return None

    colors = random.sample(range(1, 10), len(object_cells))
    masks = random.sample(_MASKS_3X3, len(object_cells))
    for (br, bc), color, mask in zip(object_cells, colors, masks):
        center_r = margin_top + 1 + 4 * br
        center_c = margin_left + 1 + 4 * bc
        _draw_object(grid, center_r, center_c, color, mask)

    comps = _components(grid)
    if len(comps) != len(object_cells):
        return None

    neighbor_counts = []
    for i, comp in enumerate(comps):
        count = 0
        for j, other in enumerate(comps):
            if i != j and _are_bbox_neighbors(comp["bbox"], other["bbox"]):
                count += 1
        neighbor_counts.append(count)
    if 0 not in neighbor_counts or 1 not in neighbor_counts or max(neighbor_counts) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
