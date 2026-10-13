from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(grid != BLACK):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        color = int(grid[sr, sc])
        comp_id = len(comps)
        cells: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        labels[sr, sc] = comp_id

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(grid[rr, cc]) == color
                ):
                    seen[rr, cc] = True
                    labels[rr, cc] = comp_id
                    q.append((rr, cc))

        coords = np.array(cells, dtype=int)
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        comps.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )

    return comps, labels


def _seam_counts(grid: np.ndarray, comps: list[dict], labels: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    seams: dict[tuple[int, int], int] = defaultdict(int)

    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a < 0:
                continue
            if r + 1 < h:
                b = int(labels[r + 1, c])
                if b >= 0 and b != a and comps[a]["color"] != comps[b]["color"]:
                    seams[tuple(sorted((a, b)))] += 1
            if c + 1 < w:
                b = int(labels[r, c + 1])
                if b >= 0 and b != a and comps[a]["color"] != comps[b]["color"]:
                    seams[tuple(sorted((a, b)))] += 1

    return dict(seams)


def _longest_seam_pair(seams: dict[tuple[int, int], int]) -> tuple[int, int]:
    assert seams, "expected at least one adjacent pair of foreground regions"
    longest = max(seams.values())
    winners = [pair for pair, length in seams.items() if length == longest]
    assert len(winners) == 1, "expected a unique longest seam"
    return winners[0]


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    r = r0
    c = c0
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a maroon center line between the two foreground regions with the longest shared seam.

    Input structure:
    - The background is Black.
    - Several non-Black colors each form one large 4-connected foreground region.
    - Some differently colored regions are 4-adjacent, and each adjacent pair may share one or more boundary segments.

    Rule:
    - Count every 4-neighbor edge where two different foreground regions touch; summing these edges gives that pair's seam length.
    - Select the unique pair of regions with the largest total seam length.
    - Draw a Maroon Bresenham line between the bounding-box centers of the selected regions, coloring only Black cells on that path.
    - Preserve every existing foreground pixel and leave all other pixels unchanged.

    Key insight:
    The selected pair is determined by a hidden boundary count, not by color, area, or absolute position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps, labels = _components(g)
    assert len(comps) >= 2, "expected at least two foreground regions"
    pair = _longest_seam_pair(_seam_counts(g, comps, labels))

    out = g.copy()
    start = comps[pair[0]]["center"]
    end = comps[pair[1]]["center"]
    for r, c in _bresenham(start, end):
        if int(out[r, c]) == BLACK:
            out[r, c] = MAROON
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, h, w = rect
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _frame_connected_after_gap(
    rect: tuple[int, int, int, int],
    side: str,
    start: int,
    length: int,
) -> bool:
    h, w = rect[2], rect[3]
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True

    if side == "right":
        mask[start : start + length, w - 1] = False
    elif side == "left":
        mask[start : start + length, 0] = False
    elif side == "bottom":
        mask[h - 1, start : start + length] = False
    else:
        mask[0, start : start + length] = False

    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    q: deque[tuple[int, int]] = deque([(int(pts[0, 0]), int(pts[0, 1]))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _clear_target_gap(
    grid: np.ndarray,
    left_rect: tuple[int, int, int, int],
    right_rect: tuple[int, int, int, int],
    gap_start: int,
    gap_len: int,
) -> None:
    if gap_len <= 0:
        return
    lr, lc, _lh, lw = left_rect
    rr, rc, _rh, _rw = right_rect
    for off in range(gap_start, gap_start + gap_len):
        grid[lr + off, lc + lw - 1] = BLACK
        grid[rr + off, rc] = BLACK


def _candidate_frames(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    max_contact: int,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
    lr, lc, lh, lw = left
    rr, rc, rh, rw = right
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []

    for base in (left, right):
        br, bc, bh, bw = base
        max_w = min(bw, max_contact)
        if max_w >= 5:
            width = random.randint(5, max_w)
            height = random.randint(5, 8)
            offset = random.randint(0, bw - width)
            candidates.append(((br - height, bc + offset, height, width), base[:2]))

            width = random.randint(5, max_w)
            height = random.randint(5, 8)
            offset = random.randint(0, bw - width)
            candidates.append(((br + bh, bc + offset, height, width), base[:2]))

        max_h = min(bh, max_contact)
        if max_h >= 5:
            height = random.randint(5, max_h)
            width = random.randint(5, 8)
            offset = random.randint(0, bh - height)
            if base is left:
                candidates.append(((br + offset, bc - width, height, width), base[:2]))
            else:
                candidates.append(((br + offset, bc + bw, height, width), base[:2]))

    random.shuffle(candidates)
    return candidates


def _normalize_rects(rects: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int, int, int]], int, int]:
    min_r = min(r for r, _c, _h, _w in rects)
    min_c = min(c for _r, c, _h, _w in rects)
    max_r = max(r + h for r, _c, h, _w in rects)
    max_c = max(c + w for _r, c, _h, w in rects)
    need_h = max_r - min_r
    need_w = max_c - min_c
    if need_h + 2 > 30 or need_w + 2 > 30:
        raise ValueError("layout too large")

    grid_h = random.randint(need_h + 2, min(30, need_h + 7))
    grid_w = random.randint(need_w + 2, min(30, need_w + 7))
    off_r = random.randint(1, grid_h - need_h - 1) - min_r
    off_c = random.randint(1, grid_w - need_w - 1) - min_c
    shifted = [(r + off_r, c + off_c, h, w) for r, c, h, w in rects]
    return shifted, grid_h, grid_w


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randint(0, 3)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    return out


def _valid_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    if MAROON in colors or len(colors) < 3:
        return False
    comps, labels = _components(grid)
    if len(comps) != len(colors):
        return False
    if any(len(comp["coords"]) < 14 for comp in comps):
        return False

    seams = _seam_counts(grid, comps, labels)
    if len(seams) < 3:
        return False
    lengths = sorted(seams.values(), reverse=True)
    if len(lengths) < 2 or lengths[0] < lengths[1] + 2:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    changed = (out != grid) & (grid == BLACK)
    if int(changed.sum()) < 2:
        return False
    return bool(np.all(out[grid != BLACK] == grid[grid != BLACK]))


def _gen_attempt() -> np.ndarray | None:
    target_h = random.randint(10, 15)
    left_w = random.randint(6, 9)
    right_w = random.randint(6, 9)
    left = (0, 0, target_h, left_w)
    right = (0, left_w, target_h, right_w)

    gap_len = 0
    gap_start = 0
    if random.random() < 0.5 and target_h >= 12:
        gap_len = random.randint(2, 4)
        mid = target_h // 2
        low = max(2, mid - gap_len + 1)
        high = min(mid, target_h - gap_len - 2)
        if low <= high:
            gap_start = random.randint(low, high)
        else:
            gap_len = 0

    if gap_len and not (
        _frame_connected_after_gap(left, "right", gap_start, gap_len)
        and _frame_connected_after_gap(right, "left", gap_start, gap_len)
    ):
        return None

    target_seam = target_h - gap_len
    max_contact = min(7, target_seam - 2)
    if max_contact < 5:
        return None

    rects = [left, right]
    wanted_extra = random.randint(2, 4)
    for cand, _parent_anchor in _candidate_frames(left, right, max_contact):
        if len(rects) >= 2 + wanted_extra:
            break
        if any(_rects_overlap(cand, existing) for existing in rects):
            continue
        rects.append(cand)

    if len(rects) < 4:
        return None

    try:
        shifted, grid_h, grid_w = _normalize_rects(rects)
    except ValueError:
        return None

    colors = random.sample(PALETTE, len(shifted))
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for rect, color in zip(shifted, colors):
        _draw_frame(grid, rect, color)

    shifted_left = shifted[0]
    shifted_right = shifted[1]
    _clear_target_gap(grid, shifted_left, shifted_right, gap_start, gap_len)

    grid = _random_transform(grid)
    if not _valid_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
