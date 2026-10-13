from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 30
INTERIOR_MIN = 2
INTERIOR_MAX = 5
PATTERN_RETRIES = 80
OBJECT_RETRIES = 240
GEN_RETRIES = 400


class FrameInfo(NamedTuple):
    bbox: tuple[int, int, int, int]
    side: str
    gap_width: int
    border_color: int


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, np.asarray(pts, dtype=int)))
    return comps


def _gap_info(side_mask: np.ndarray) -> tuple[int, np.ndarray] | None:
    gap_idx = np.flatnonzero(~side_mask)
    if gap_idx.size == 0 or gap_idx.size > 3:
        return None
    if gap_idx[0] == 0 or gap_idx[-1] == len(side_mask) - 1:
        return None
    if gap_idx[-1] - gap_idx[0] + 1 != gap_idx.size:
        return None
    return int(gap_idx.size), gap_idx


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> FrameInfo | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(local[1:-1, 1:-1]):
        return None

    top = local[0, :]
    bottom = local[-1, :]
    left = local[:, 0]
    right = local[:, -1]
    sides = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }

    missing = {name: len(mask) - int(mask.sum()) for name, mask in sides.items()}
    gap_sides = [name for name, count in missing.items() if count > 0]
    if len(gap_sides) != 1:
        return None
    side = gap_sides[0]
    if any(missing[name] != 0 for name in sides if name != side):
        return None

    gap = _gap_info(sides[side])
    if gap is None:
        return None
    gap_width, gap_idx = gap

    perimeter = 2 * h + 2 * w - 4
    if len(coords) != perimeter - gap_width:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if side == "top":
        if not np.all(patch[0, gap_idx] == bg):
            return None
    elif side == "bottom":
        if not np.all(patch[-1, gap_idx] == bg):
            return None
    elif side == "left":
        if not np.all(patch[gap_idx, 0] == bg):
            return None
    else:
        if not np.all(patch[gap_idx, -1] == bg):
            return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any((interior != bg) & (interior != color)):
        return None

    return FrameInfo((r0, c0, r1, c1), side, gap_width, int(color))


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    view = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each framed interior pattern toward the grid edge indicated by the frame gap.

    Input structure:
    - The background is black.
    - Several non-overlapping 1-pixel rectangular frames appear on the grid, each in its own border color.
    - Every frame encloses a small pattern in a different color, and exactly one side of the frame has a contiguous gap of width 1 to 3.

    Rule:
    - Detect which side of each frame contains the only gap: top, bottom, left, or right.
    - Use that side as the projection direction, and use the gap width G as the distance from the copied pattern to the corresponding grid edge.
    - Copy the non-background interior pattern in that direction so the copied pattern's bounding box is G cells from the target edge, while keeping the perpendicular coordinate unchanged.
    - Leave the original frames, gaps, and interior patterns unchanged.

    Key insight:
    - The gap encodes two independent signals at once: its side gives the direction, and its width gives the exact offset from the edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    frames: list[FrameInfo] = []
    for color, coords in _components(g, bg):
        info = _frame_info(g, color, coords, bg)
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: item.bbox)

    for info in frames:
        r0, c0, r1, c1 = info.bbox
        patch = g[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere(patch != bg)
        if pts.size == 0:
            continue
        pr0, pc0 = pts.min(axis=0)
        pr1, pc1 = pts.max(axis=0)
        sub = patch[pr0 : pr1 + 1, pc0 : pc1 + 1]
        ph, pw = sub.shape
        if info.side == "top":
            top, left = info.gap_width, c0 + 1 + int(pc0)
        elif info.side == "bottom":
            top, left = g.shape[0] - info.gap_width - ph, c0 + 1 + int(pc0)
        elif info.side == "left":
            top, left = r0 + 1 + int(pr0), info.gap_width
        else:
            top, left = r0 + 1 + int(pr0), g.shape[1] - info.gap_width - pw
        _paste_non_bg(out, sub, top, left, bg)

    return out


def _random_path(a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    cells = [(r0, c0)]
    if rng.random() < 0.5:
        step = 1 if r1 >= r0 else -1
        for rr in range(r0 + step, r1 + step, step):
            cells.append((rr, c0))
        step = 1 if c1 >= c0 else -1
        for cc in range(c0 + step, c1 + step, step):
            cells.append((r1, cc))
    else:
        step = 1 if c1 >= c0 else -1
        for cc in range(c0 + step, c1 + step, step):
            cells.append((r0, cc))
        step = 1 if r1 >= r0 else -1
        for rr in range(r0 + step, r1 + step, step):
            cells.append((rr, c1))
    return cells


def _pattern_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    area = height * width
    for _ in range(PATTERN_RETRIES):
        anchors = [
            (0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1),
        ]
        rng.shuffle(anchors)
        cells = {anchors[0]}
        current = anchors[0]
        for target in anchors[1:]:
            cells.update(_random_path(current, target, rng))
            current = target

        lo = max(len(cells), max(1, int(np.ceil(0.35 * area))))
        hi = max(lo, min(area, int(np.ceil(0.8 * area))))
        target_size = int(rng.integers(lo, hi + 1))
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        mask = np.zeros((height, width), dtype=bool)
        pts = np.asarray(sorted(cells), dtype=int)
        mask[pts[:, 0], pts[:, 1]] = True
        if mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
            return mask
    raise RuntimeError("pattern generation failed")


def _expanded(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _clear(reserved: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded(box, pad, reserved.shape)
    return not reserved[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve(reserved: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = _expanded(box, pad, reserved.shape)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _origin_ranges(grid_h: int, grid_w: int, side: str, gap: int, inner_h: int, inner_w: int):
    r_lo = 1
    r_hi = grid_h - inner_h - 3
    c_lo = 1
    c_hi = grid_w - inner_w - 3
    if side == "top":
        r_lo = max(r_lo, gap + inner_h + 1)
    elif side == "bottom":
        r_hi = min(r_hi, grid_h - gap - 2 * inner_h - 3)
    elif side == "left":
        c_lo = max(c_lo, gap + inner_w + 1)
    else:
        c_hi = min(c_hi, grid_w - gap - 2 * inner_w - 3)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _copy_box(
    grid_h: int,
    grid_w: int,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    gap: int,
) -> tuple[int, int, int, int]:
    if side == "top":
        tr, tc = gap, c0 + 1
    elif side == "bottom":
        tr, tc = grid_h - gap - inner_h, c0 + 1
    elif side == "left":
        tr, tc = r0 + 1, gap
    else:
        tr, tc = r0 + 1, grid_w - gap - inner_w
    return tr, tc, tr + inner_h - 1, tc + inner_w - 1


def _draw_object(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    border_color: int,
    pattern_color: int,
    side: str,
    gap: int,
    gap_offset: int,
    pattern: np.ndarray,
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    if side == "top":
        grid[r0, c0 + 1 + gap_offset : c0 + 1 + gap_offset + gap] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_offset : c0 + 1 + gap_offset + gap] = BG
    elif side == "left":
        grid[r0 + 1 + gap_offset : r0 + 1 + gap_offset + gap, c0] = BG
    else:
        grid[r0 + 1 + gap_offset : r0 + 1 + gap_offset + gap, c1] = BG
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[pattern] = pattern_color


def _place_one(
    grid: np.ndarray,
    reserved: np.ndarray,
    side: str,
    border_color: int,
    pattern_color: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], np.ndarray, int, int] | None:
    grid_h, grid_w = grid.shape
    for _ in range(OBJECT_RETRIES):
        if side in ("top", "bottom"):
            inner_h = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
            inner_w = int(rng.integers(max(INTERIOR_MIN, 1), INTERIOR_MAX + 1))
            gap = int(rng.integers(1, min(3, inner_w) + 1))
        else:
            inner_h = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
            inner_w = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
            gap = int(rng.integers(1, min(3, inner_h) + 1))

        ranges = _origin_ranges(grid_h, grid_w, side, gap, inner_h, inner_w)
        if ranges is None:
            continue
        r_lo, r_hi, c_lo, c_hi = ranges
        r0 = int(rng.integers(r_lo, r_hi + 1))
        c0 = int(rng.integers(c_lo, c_hi + 1))
        orig_box = (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1)
        copy_box = _copy_box(grid_h, grid_w, r0, c0, inner_h, inner_w, side, gap)
        if _overlap(orig_box, copy_box):
            continue
        if not _clear(reserved, orig_box, pad=1) or not _clear(reserved, copy_box, pad=1):
            continue

        pattern = _pattern_mask(inner_h, inner_w, rng)
        gap_span = inner_w if side in ("top", "bottom") else inner_h
        gap_offset = int(rng.integers(0, gap_span - gap + 1))

        _draw_object(
            grid,
            r0,
            c0,
            inner_h,
            inner_w,
            border_color,
            pattern_color,
            side,
            gap,
            gap_offset,
            pattern,
        )
        _reserve(reserved, orig_box, pad=1)
        _reserve(reserved, copy_box, pad=1)
        return copy_box, pattern.copy(), pattern_color, gap
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid_w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    max_objects = 2 if min(grid_h, grid_w) < 22 else 3 if min(grid_h, grid_w) < 26 else 4
    n_objects = int(rng.integers(1, max_objects + 1))
    directions = list(rng.permutation(np.array(SIDES, dtype=object))[:n_objects])
    chosen_colors = [int(c) for c in rng.choice(COLORS, size=2 * n_objects, replace=False)]

    copies: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []
    for idx, side in enumerate(directions):
        border_color = chosen_colors[2 * idx]
        pattern_color = chosen_colors[2 * idx + 1]
        placed = _place_one(grid, reserved, str(side), border_color, pattern_color, rng)
        if placed is None:
            return None
        copy_box, pattern, color, _gap = placed
        copies.append((copy_box, pattern, color))

    out = grid.copy()
    for (r0, c0, r1, c1), pattern, color in copies:
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[pattern] = color
        _paste_non_bg(out, patch, r0, c0, BG)

    if np.array_equal(out, grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
