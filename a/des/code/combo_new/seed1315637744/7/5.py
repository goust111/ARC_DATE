import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
ORANGE = 7
PORTAL_COLORS = [1, 2, 5, 6, 8, 9]
GEN_RETRIES = 300
PAIR_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _teleport_exit(
    src_bbox: tuple[int, int, int, int],
    dst_bbox: tuple[int, int, int, int],
    hit_r: int,
    hit_c: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = src_bbox
    dr0, dc0, dr1, dc1 = dst_bbox
    if dr == 0 and dc == 1:
        assert hit_c == sc0
        return dr0 + (hit_r - sr0), dc1
    if dr == 0 and dc == -1:
        assert hit_c == sc1
        return dr0 + (hit_r - sr0), dc0
    if dr == 1 and dc == 0:
        assert hit_r == sr0
        return dr1, dc0 + (hit_c - sc0)
    assert dr == -1 and dc == 0
    assert hit_r == sr1
    return dr0, dc0 + (hit_c - sc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a cardinal ray through paired solid-color portal rectangles and continue it from the matched exit.

    Input structure:
    - The background is black.
    - There are several solid monochrome rectangles; each non-marker color appears in exactly two same-sized rectangles.
    - One yellow singleton marks the ray source, and one adjacent green singleton marks the initial horizontal or vertical direction.

    Rule:
    - Fire a ray from the yellow cell in the direction indicated by the green cell, recoloring only black traversed cells to orange.
    - If the ray reaches a colored rectangle, do not recolor the rectangle; instead, match it with the other rectangle of the same color and size.
    - Reappear at the opposite face of that partner rectangle at the corresponding offset, keep the same direction, and continue until the ray leaves the grid.

    Key insight:
    - Each same-color rectangle pair is a portal: entering one face maps to the opposite face of its twin while preserving the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    start_cells = np.argwhere(g == YELLOW)
    dir_cells = np.argwhere(g == GREEN)
    assert start_cells.shape[0] == 1
    assert dir_cells.shape[0] == 1
    sr, sc = map(int, start_cells[0])
    gr, gc = map(int, dir_cells[0])
    dr, dc = gr - sr, gc - sc
    assert abs(dr) + abs(dc) == 1

    portal_colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, GREEN, YELLOW)]
    rects: list[dict[str, object]] = []
    pair_of: dict[int, int] = {}
    cell_to_rect: dict[tuple[int, int], int] = {}

    for color in portal_colors:
        comps = _components(g == color)
        assert len(comps) == 2
        ids: list[int] = []
        sizes: list[tuple[int, int]] = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1)
            rid = len(rects)
            rects.append({"color": color, "bbox": (r0, c0, r1, c1)})
            ids.append(rid)
            sizes.append((r1 - r0 + 1, c1 - c0 + 1))
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    cell_to_rect[(rr, cc)] = rid
        assert sizes[0] == sizes[1]
        pair_of[ids[0]] = ids[1]
        pair_of[ids[1]] = ids[0]

    r, c = sr, sc
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        rid = cell_to_rect.get((nr, nc))
        if rid is not None:
            pair_id = pair_of[rid]
            src_bbox = rects[rid]["bbox"]
            dst_bbox = rects[pair_id]["bbox"]
            assert isinstance(src_bbox, tuple) and isinstance(dst_bbox, tuple)
            r, c = _teleport_exit(src_bbox, dst_bbox, nr, nc, dr, dc)
            continue

        if int(g[nr, nc]) == BLACK:
            out[nr, nc] = ORANGE
        r, c = nr, nc

    return out


def _can_place_rect(occupied: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> bool:
    h, w = occupied.shape
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _reserve_rect(occupied: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> None:
    occupied[r0 : r0 + rh, c0 : c0 + rw] = True


def _random_shape() -> tuple[int, int]:
    while True:
        rh = random.randint(1, 4)
        rw = random.randint(1, 4)
        if rh * rw >= 2:
            return rh, rw


def _apply_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        return grid.copy()
    if transform_id == 1:
        return np.flipud(grid)
    if transform_id == 2:
        return np.fliplr(grid)
    if transform_id == 3:
        return np.flipud(np.fliplr(grid))
    if transform_id == 4:
        return grid.T.copy()
    if transform_id == 5:
        return np.flipud(grid.T)
    if transform_id == 6:
        return np.fliplr(grid.T)
    return np.flipud(np.fliplr(grid.T))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(16, 28)
    used_pairs = random.randint(0, 3)
    min_extra = 1 if used_pairs == 0 else 0
    max_extra = min(2, len(PORTAL_COLORS) - used_pairs)
    extra_pairs = random.randint(min_extra, max_extra)
    total_pairs = used_pairs + extra_pairs
    colors = random.sample(PORTAL_COLORS, total_pairs)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    grid[0, 0] = YELLOW
    grid[0, 1] = GREEN
    occupied[0, 0] = True
    occupied[0, 1] = True

    used_layout: list[dict[str, object]] = []
    current_row = 0
    current_pos = 1

    for idx in range(used_pairs):
        color = colors[idx]
        placed = False
        for _ in range(PAIR_RETRIES):
            rh, rw = _random_shape()
            entry_offset_min = max(0, current_row - (h - rh))
            entry_offset_max = min(rh - 1, current_row)
            if entry_offset_min > entry_offset_max:
                continue
            entry_offset = random.randint(entry_offset_min, entry_offset_max)
            a_r0 = current_row - entry_offset

            new_row_min = entry_offset
            new_row_max = h - rh + entry_offset
            new_row = random.randint(new_row_min, new_row_max)
            b_r0 = new_row - entry_offset

            min_future = (used_pairs - idx - 1) * 7 + 2
            a_c0_min = current_pos + 2
            a_c0_max = w - (2 * rw + 3 + min_future)
            if a_c0_min > a_c0_max:
                continue
            a_c0 = random.randint(a_c0_min, a_c0_max)

            b_c0_min = a_c0 + rw + 1
            b_c0_max = w - (rw + 1 + min_future)
            if b_c0_min > b_c0_max:
                continue
            b_c0 = random.randint(b_c0_min, b_c0_max)

            if not _can_place_rect(occupied, a_r0, a_c0, rh, rw):
                continue
            if not _can_place_rect(occupied, b_r0, b_c0, rh, rw):
                continue

            _reserve_rect(occupied, a_r0, a_c0, rh, rw)
            _reserve_rect(occupied, b_r0, b_c0, rh, rw)
            grid[a_r0 : a_r0 + rh, a_c0 : a_c0 + rw] = color
            grid[b_r0 : b_r0 + rh, b_c0 : b_c0 + rw] = color
            used_layout.append(
                {
                    "entry_row": current_row,
                    "before_pos": current_pos,
                    "entry": (a_r0, a_c0, rh, rw),
                    "exit": (b_r0, b_c0, rh, rw),
                    "exit_row": new_row,
                }
            )
            current_row = new_row
            current_pos = b_c0 + rw - 1
            placed = True
            break
        if not placed:
            return None

    if current_pos >= w - 1:
        return None

    path_mask = np.zeros((h, w), dtype=bool)
    path_mask[0, 0] = True
    path_mask[0, 1] = True
    trace_row = 0
    trace_pos = 1
    for item in used_layout:
        _a_r0, a_c0, _rh, _rw = item["entry"]
        if trace_pos + 1 < a_c0:
            path_mask[trace_row, trace_pos + 1 : a_c0] = True
        trace_row = int(item["exit_row"])
        _b_r0, b_c0, _rh2, rw2 = item["exit"]
        trace_pos = b_c0 + rw2 - 1
    path_mask[trace_row, trace_pos + 1 : w] = True

    for color in colors[used_pairs:]:
        placed = False
        for _ in range(PAIR_RETRIES):
            rh, rw = _random_shape()
            first = None
            second = None
            for _ in range(PAIR_RETRIES):
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                if not _can_place_rect(occupied, r0, c0, rh, rw):
                    continue
                if path_mask[r0 : r0 + rh, c0 : c0 + rw].any():
                    continue
                first = (r0, c0)
                break
            if first is None:
                continue
            _reserve_rect(occupied, first[0], first[1], rh, rw)
            grid[first[0] : first[0] + rh, first[1] : first[1] + rw] = color

            for _ in range(PAIR_RETRIES):
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                if not _can_place_rect(occupied, r0, c0, rh, rw):
                    continue
                if path_mask[r0 : r0 + rh, c0 : c0 + rw].any():
                    continue
                second = (r0, c0)
                break
            if second is not None:
                _reserve_rect(occupied, second[0], second[1], rh, rw)
                grid[second[0] : second[0] + rh, second[1] : second[1] + rw] = color
                placed = True
                break

            grid[first[0] : first[0] + rh, first[1] : first[1] + rw] = BLACK
            occupied[first[0] : first[0] + rh, first[1] : first[1] + rw] = False
        if not placed:
            return None

    transformed = _apply_transform(grid, random.randint(0, 7))
    _ = main(transformed)
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
