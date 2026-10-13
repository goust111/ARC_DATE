from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREY = 5
        TEAL = 8

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREY = Color.GREY
TEAL = Color.TEAL
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rect_from_markers(points: np.ndarray) -> tuple[int, int, int, int]:
    assert points.shape == (2, 2), "expected exactly two marker pixels"
    rows = sorted(int(r) for r in points[:, 0])
    cols = sorted(int(c) for c in points[:, 1])
    assert rows[0] < rows[1] and cols[0] < cols[1], "marker pair must be diagonal"
    return rows[0], rows[1], cols[0], cols[1]


def _segments(size: int, lines: list[int]) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = 0
    for line in sorted(set(int(x) for x in lines)):
        if start <= line - 1:
            segs.append((int(start), int(line - 1)))
        start = int(line + 1)
    if start <= int(size - 1):
        segs.append((int(start), int(size - 1)))
    return segs


def _region_inside_rect(
    row_seg: tuple[int, int],
    col_seg: tuple[int, int],
    rect: tuple[int, int, int, int],
) -> bool:
    r0, r1 = map(int, row_seg)
    c0, c1 = map(int, col_seg)
    rr0, rr1, cc0, cc1 = map(int, rect)
    return rr0 < r0 and r1 < rr1 and cc0 < c0 and c1 < cc1


def _cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == TEAL):
        rr = int(r)
        cc = int(c)
        if all(in_bounds(grid.shape, rr + dr, cc + dc) and int(grid[rr + dr, cc + dc]) == TEAL for dr, dc in DIRS4):
            centers.append((rr, cc))
    return centers


def _line_output(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    red_pts = np.argwhere(g == RED)
    blue_pts = np.argwhere(g == BLUE)
    red_rect = _rect_from_markers(red_pts)
    blue_rect = _rect_from_markers(blue_pts)

    out = g.copy()
    red_rows = [red_rect[0], red_rect[1]]
    red_cols = [red_rect[2], red_rect[3]]
    blue_rows = [blue_rect[0], blue_rect[1]]
    blue_cols = [blue_rect[2], blue_rect[3]]

    out[red_rows, :] = RED
    out[:, red_cols] = RED
    out[blue_rows, :] = BLUE
    out[:, blue_cols] = BLUE
    return out, red_rect, blue_rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw full-row/full-column dividers from two marker pairs, then fill only the regions activated by cyan cross centers.

    Input structure:
    - The background is black.
    - There are exactly two red marker pixels and two blue marker pixels, with each same-color pair on different rows and columns.
    - Cyan plus-shaped crosses and isolated grey noise pixels are scattered away from the markers.

    Rule:
    - Treat each same-color marker pair as opposite rectangle corners, then paint their two rows and two columns across the full grid: red first, blue second.
    - These divider lines partition the non-line cells into rectangular regions; detect cyan cross centers in the original input before any lines are drawn.
    - For every region containing at least one such center, fill only its remaining black cells with blue if the region lies strictly inside the blue rectangle, otherwise red.

    Key insight:
    - The markers define the partition geometry, but the cyan cross centers decide which partitions become visible through region-local filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    centers = _cross_centers(g)
    out, red_rect, blue_rect = _line_output(g)

    row_lines = [red_rect[0], red_rect[1], blue_rect[0], blue_rect[1]]
    col_lines = [red_rect[2], red_rect[3], blue_rect[2], blue_rect[3]]
    row_segs = _segments(g.shape[0], row_lines)
    col_segs = _segments(g.shape[1], col_lines)

    for row_seg in row_segs:
        for col_seg in col_segs:
            r0, r1 = map(int, row_seg)
            c0, c1 = map(int, col_seg)
            active = any(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in centers)
            if not active:
                continue
            fill_color = BLUE if _region_inside_rect(row_seg, col_seg, blue_rect) else RED
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[patch == BG] = int(fill_color)
    return out


LAYOUT_PATTERNS = (
    ("nested", (0, 3), (0, 3), (1, 2), (1, 2)),
    ("partial", (0, 2), (0, 2), (1, 3), (1, 3)),
    ("offset_h", (0, 3), (0, 2), (1, 2), (1, 3)),
    ("offset_v", (0, 2), (0, 3), (1, 3), (1, 2)),
)
GEN_RETRIES = 400


def _bands_to_lines(bands: list[int]) -> list[int]:
    lines: list[int] = []
    acc = int(bands[0])
    lines.append(acc)
    for band in bands[1:-1]:
        acc += 1 + int(band)
        lines.append(acc)
    assert len(lines) == 4
    return lines


def _sample_bands() -> list[int]:
    return [random.randint(2, 5) for _ in range(5)]


def _marker_pixels(
    row_lines: list[int],
    col_lines: list[int],
    row_pick: tuple[int, int],
    col_pick: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0 = int(row_lines[row_pick[0]])
    r1 = int(row_lines[row_pick[1]])
    c0 = int(col_lines[col_pick[0]])
    c1 = int(col_lines[col_pick[1]])
    if random.random() < 0.5:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _cross_pixels(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _block_pixels(blocked: np.ndarray, pixels: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in pixels:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                blocked[nr, nc] = True


def _can_place_pixels(blocked: np.ndarray, pixels: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in pixels:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if blocked[rr, cc]:
            return False
    return True


def _region_specs(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
    red_rect: tuple[int, int, int, int],
    blue_rect: tuple[int, int, int, int],
    marker_set: set[tuple[int, int]],
) -> list[dict]:
    h, w = map(int, shape)
    specs: list[dict] = []
    for row_seg in _segments(h, row_lines):
        for col_seg in _segments(w, col_lines):
            r0, r1 = map(int, row_seg)
            c0, c1 = map(int, col_seg)
            centers: list[tuple[int, int]] = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if not (0 < r < h - 1 and 0 < c < w - 1):
                        continue
                    pixels = _cross_pixels((r, c))
                    if any((pr, pc) in marker_set for pr, pc in pixels):
                        continue
                    centers.append((int(r), int(c)))
            specs.append(
                {
                    "row_seg": row_seg,
                    "col_seg": col_seg,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    "fill": BLUE if _region_inside_rect(row_seg, col_seg, blue_rect) else RED,
                    "centers": centers,
                }
            )
    return specs


def _choose_active_regions(regions: list[dict], n_crosses: int) -> dict[int, int] | None:
    available = [i for i, region in enumerate(regions) if region["centers"]]
    if len(available) < 2:
        return None
    red_regions = [i for i in available if int(regions[i]["fill"]) == RED]
    blue_regions = [i for i in available if int(regions[i]["fill"]) == BLUE]
    if not red_regions:
        return None

    n_active = random.randint(2, min(4, n_crosses, len(available) - 1))
    active: set[int] = {random.choice(red_regions)}
    if blue_regions and random.random() < 0.7:
        active.add(random.choice(blue_regions))

    pool = available[:]
    random.shuffle(pool)
    for idx in pool:
        if len(active) >= n_active:
            break
        active.add(int(idx))
    if len(active) >= len(regions):
        return None

    weights = [max(1, int(regions[idx]["area"])) for idx in active]
    allocations = {idx: 1 for idx in active}
    active_list = list(active)
    for _ in range(n_crosses - len(active_list)):
        chosen = random.choices(active_list, weights=weights, k=1)[0]
        allocations[int(chosen)] += 1
    return allocations


def _place_crosses(grid: np.ndarray, blocked: np.ndarray, regions: list[dict], allocations: dict[int, int]) -> bool:
    order = sorted(allocations, key=lambda idx: int(regions[idx]["area"]), reverse=True)
    for idx in order:
        region = regions[int(idx)]
        for _ in range(int(allocations[idx])):
            centers = region["centers"][:]
            if not centers:
                return False
            if random.random() < 0.5:
                centers.sort(
                    key=lambda rc: min(
                        rc[0] - int(region["row_seg"][0]),
                        int(region["row_seg"][1]) - rc[0],
                        rc[1] - int(region["col_seg"][0]),
                        int(region["col_seg"][1]) - rc[1],
                    )
                )
            else:
                random.shuffle(centers)
            placed = False
            for center in centers:
                pixels = _cross_pixels(center)
                if not _can_place_pixels(blocked, pixels):
                    continue
                for r, c in pixels:
                    grid[int(r), int(c)] = TEAL
                _block_pixels(blocked, pixels)
                placed = True
                break
            if not placed:
                return False
    return True


def _place_noise(grid: np.ndarray, blocked: np.ndarray, row_lines: list[int], col_lines: list[int]) -> bool:
    h, w = grid.shape
    line_cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(r) in set(row_lines) or int(c) in set(col_lines)
    }
    n_noise = random.randint(2, 6)
    for _ in range(n_noise):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not blocked[r, c] and (0 < r < h - 1 or 0 < c < w - 1)
        ]
        if not candidates:
            return False
        if random.random() < 0.5:
            line_candidates = [pt for pt in candidates if pt in line_cells]
            if line_candidates:
                candidates = line_candidates
        cell = random.choice(candidates)
        grid[cell] = GREY
        _block_pixels(blocked, [cell])
    return True


def _count_crosses(grid: np.ndarray) -> int:
    return len(_cross_centers(grid))


def _gen_attempt() -> np.ndarray | None:
    row_bands = _sample_bands()
    col_bands = _sample_bands()
    h = int(sum(row_bands) + 4)
    w = int(sum(col_bands) + 4)
    row_lines = _bands_to_lines(row_bands)
    col_lines = _bands_to_lines(col_bands)
    _name, red_rows, red_cols, blue_rows, blue_cols = random.choice(LAYOUT_PATTERNS)

    red_markers = _marker_pixels(row_lines, col_lines, red_rows, red_cols)
    blue_markers = _marker_pixels(row_lines, col_lines, blue_rows, blue_cols)
    marker_set = {tuple(map(int, rc)) for rc in red_markers + blue_markers}
    if len(marker_set) != 4:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in red_markers:
        grid[int(r), int(c)] = RED
    for r, c in blue_markers:
        grid[int(r), int(c)] = BLUE
    _block_pixels(blocked, list(marker_set))

    red_rect = _rect_from_markers(np.array(red_markers, dtype=int))
    blue_rect = _rect_from_markers(np.array(blue_markers, dtype=int))
    regions = _region_specs(grid.shape, row_lines, col_lines, red_rect, blue_rect, marker_set)

    n_crosses = random.randint(4, 8)
    allocations = _choose_active_regions(regions, n_crosses)
    if allocations is None:
        return None
    if not _place_crosses(grid, blocked, regions, allocations):
        return None
    if _count_crosses(grid) != n_crosses:
        return None
    if not _place_noise(grid, blocked, row_lines, col_lines):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None

    active_region_count = 0
    for region in regions:
        r0, r1 = map(int, region["row_seg"])
        c0, c1 = map(int, region["col_seg"])
        if any(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in _cross_centers(grid)):
            active_region_count += 1
    if active_region_count < 1 or active_region_count >= len(regions):
        return None

    counts = Counter(int(v) for v in grid.flatten())
    if counts[RED] != 2 or counts[BLUE] != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
