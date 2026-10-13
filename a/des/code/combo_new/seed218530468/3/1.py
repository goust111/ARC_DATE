import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 7)

GEN_RETRIES = 400
FRAME_PLACEMENT_TRIES = 120
RESERVE_PAD = 2
FRAME_COUNT_OPTIONS = (1, 2, 3)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_frame_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    pts = set(comp)
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if pts != expected:
        return None
    return r0, r1, c0, c1


def _quadrant_rows_cols(r0: int, r1: int, c0: int, c1: int) -> dict[str, tuple[list[int], list[int]]]:
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    inside_rows = list(range(r0 + 1, r1))
    inside_cols = list(range(c0 + 1, c1))
    top_rows = [r for r in inside_rows if r < mid_r]
    bottom_rows = [r for r in inside_rows if r > mid_r]
    left_cols = [c for c in inside_cols if c < mid_c]
    right_cols = [c for c in inside_cols if c > mid_c]
    return {
        "tl": (top_rows, left_cols),
        "tr": (top_rows, right_cols),
        "bl": (bottom_rows, left_cols),
        "br": (bottom_rows, right_cols),
    }


def _collect_markers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[str, int, int, int]]:
    h, w = grid.shape
    r0, r1, c0, c1 = bbox
    markers: list[tuple[str, int, int, int]] = []
    if r0 > 0:
        rr = r0 - 1
        for cc in range(c0, c1 + 1):
            color = int(grid[rr, cc])
            if color not in (BG, FRAME):
                markers.append(("top", rr, cc, color))
    if r1 + 1 < h:
        rr = r1 + 1
        for cc in range(c0, c1 + 1):
            color = int(grid[rr, cc])
            if color not in (BG, FRAME):
                markers.append(("bottom", rr, cc, color))
    if c0 > 0:
        cc = c0 - 1
        for rr in range(r0, r1 + 1):
            color = int(grid[rr, cc])
            if color not in (BG, FRAME):
                markers.append(("left", rr, cc, color))
    if c1 + 1 < w:
        cc = c1 + 1
        for rr in range(r0, r1 + 1):
            color = int(grid[rr, cc])
            if color not in (BG, FRAME):
                markers.append(("right", rr, cc, color))
    return markers


def _marker_quadrant(
    side: str,
    r: int,
    c: int,
    bbox: tuple[int, int, int, int],
) -> tuple[str, int] | None:
    r0, r1, c0, c1 = bbox
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0

    if side == "top":
        if c < mid_c:
            corner = (r0, c0)
            quadrant = "tl"
        elif c > mid_c:
            corner = (r0, c1)
            quadrant = "tr"
        else:
            return None
    elif side == "bottom":
        if c < mid_c:
            corner = (r1, c0)
            quadrant = "bl"
        elif c > mid_c:
            corner = (r1, c1)
            quadrant = "br"
        else:
            return None
    elif side == "left":
        if r < mid_r:
            corner = (r0, c0)
            quadrant = "tl"
        elif r > mid_r:
            corner = (r1, c0)
            quadrant = "bl"
        else:
            return None
    else:
        if r < mid_r:
            corner = (r0, c1)
            quadrant = "tr"
        elif r > mid_r:
            corner = (r1, c1)
            quadrant = "br"
        else:
            return None

    dist = abs(r - corner[0]) + abs(c - corner[1])
    return quadrant, dist


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == FRAME):
        bbox = _rect_frame_bbox(comp)
        if bbox is not None:
            frames.append(bbox)
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey frame interior by reading which quadrant each attached border marker controls.

    Input structure:
    - The background is black.
    - One or more hollow grey rectangular frames appear in the grid as single-pixel-thick borders.
    - Single colored pixels may sit immediately outside a frame's top, bottom, left, or right edge.

    Rule:
    - For every grey frame, use its horizontal and vertical midlines to split the interior into top-left, top-right, bottom-left, and bottom-right quadrants.
    - A top or bottom marker controls the left or right quadrant on that side according to whether it lies left or right of the vertical midline; a left or right marker controls the upper or lower quadrant on that side according to whether it lies above or below the horizontal midline.
    - If multiple markers nominate the same quadrant, keep the color of the marker whose position is closer to that quadrant's corner. Erase all markers back to black, keep the grey border, and leave any unassigned quadrant black.

    Key insight:
    The marker color alone is not enough; its edge attachment and its offset relative to the frame center determine which interior quadrant it paints.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for bbox in _detect_frames(g):
        r0, r1, c0, c1 = bbox
        best: dict[str, tuple[int, int, int, int]] = {}
        markers = _collect_markers(g, bbox)
        for side, r, c, color in markers:
            out[r, c] = BG
            mapped = _marker_quadrant(side, r, c, bbox)
            if mapped is None:
                continue
            quadrant, dist = mapped
            candidate = (dist, r, c, color)
            if quadrant not in best or candidate < best[quadrant]:
                best[quadrant] = candidate

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for quadrant, (rows, cols) in _quadrant_rows_cols(r0, r1, c0, c1).items():
            if quadrant not in best or not rows or not cols:
                continue
            out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = best[quadrant][3]

    return out


def _quadrant_candidates(
    bbox: tuple[int, int, int, int],
    quadrant: str,
) -> tuple[list[tuple[tuple[int, int], int]], list[tuple[tuple[int, int], int]]]:
    r0, r1, c0, c1 = bbox
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0

    if quadrant == "tl":
        first = [((r0 - 1, c), 1 + (c - c0)) for c in range(c0 + 1, c1) if c < mid_c]
        second = [((r, c0 - 1), 1 + (r - r0)) for r in range(r0 + 1, r1) if r < mid_r]
    elif quadrant == "tr":
        first = [((r0 - 1, c), 1 + (c1 - c)) for c in range(c0 + 1, c1) if c > mid_c]
        second = [((r, c1 + 1), 1 + (r - r0)) for r in range(r0 + 1, r1) if r < mid_r]
    elif quadrant == "bl":
        first = [((r1 + 1, c), 1 + (c - c0)) for c in range(c0 + 1, c1) if c < mid_c]
        second = [((r, c0 - 1), 1 + (r1 - r)) for r in range(r0 + 1, r1) if r > mid_r]
    else:
        first = [((r1 + 1, c), 1 + (c1 - c)) for c in range(c0 + 1, c1) if c > mid_c]
        second = [((r, c1 + 1), 1 + (r1 - r)) for r in range(r0 + 1, r1) if r > mid_r]

    return first, second


def _choose_pair(
    rng: random.Random,
    first: list[tuple[tuple[int, int], int]],
    second: list[tuple[tuple[int, int], int]],
    require_distinct: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    prefer_cross = bool(first and second and rng.random() < 0.65)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if prefer_cross:
        for pos_a, dist_a in first:
            for pos_b, dist_b in second:
                if not require_distinct or dist_a != dist_b:
                    pairs.append((pos_a, pos_b))
    if not pairs:
        combined = first + second
        for i in range(len(combined)):
            for j in range(i + 1, len(combined)):
                pos_a, dist_a = combined[i]
                pos_b, dist_b = combined[j]
                if not require_distinct or dist_a != dist_b:
                    pairs.append((pos_a, pos_b))
    if not pairs:
        return None
    return rng.choice(pairs)


def _paint_quadrant_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> int:
    markers_added = 0
    patterns = ("empty", "single", "double_same", "double_conflict")

    for quadrant in ("tl", "tr", "bl", "br"):
        first, second = _quadrant_candidates(bbox, quadrant)
        combined = first + second
        if not combined:
            continue

        pattern = rng.choice(patterns)
        if pattern == "empty":
            continue

        if pattern == "single":
            pos, _dist = rng.choice(combined)
            color = rng.choice(MARKER_COLORS)
            grid[pos] = color
            markers_added += 1
            continue

        pair = _choose_pair(rng, first, second, require_distinct=(pattern == "double_conflict"))
        if pair is None:
            pos, _dist = rng.choice(combined)
            color = rng.choice(MARKER_COLORS)
            grid[pos] = color
            markers_added += 1
            continue

        pos_a, pos_b = pair
        if pattern == "double_same":
            color = rng.choice(MARKER_COLORS)
            grid[pos_a] = color
            grid[pos_b] = color
        else:
            mapped_a = _marker_quadrant(
                "top" if pos_a[0] < bbox[0] else "bottom" if pos_a[0] > bbox[1] else "left" if pos_a[1] < bbox[2] else "right",
                pos_a[0],
                pos_a[1],
                bbox,
            )
            mapped_b = _marker_quadrant(
                "top" if pos_b[0] < bbox[0] else "bottom" if pos_b[0] > bbox[1] else "left" if pos_b[1] < bbox[2] else "right",
                pos_b[0],
                pos_b[1],
                bbox,
            )
            assert mapped_a is not None and mapped_b is not None
            dist_a = mapped_a[1]
            dist_b = mapped_b[1]
            if dist_b < dist_a:
                pos_a, pos_b = pos_b, pos_a
            winner = rng.choice(MARKER_COLORS)
            loser_choices = [color for color in MARKER_COLORS if color != winner]
            loser = rng.choice(loser_choices)
            grid[pos_a] = winner
            grid[pos_b] = loser

        markers_added += 2

    return markers_added


def _frame_size_choices(limit: int) -> list[int]:
    return [size for size in range(6, min(limit, 14) + 1, 2)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_OPTIONS)
    if frame_count == 1:
        h = rng.randint(10, 30)
        w = rng.randint(10, 30)
    elif frame_count == 2:
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
    else:
        h = rng.randint(22, 30)
        w = rng.randint(22, 30)

    height_choices = _frame_size_choices(h - 4)
    width_choices = _frame_size_choices(w - 4)
    if not height_choices or not width_choices:
        return None

    if frame_count == 3:
        height_choices = [size for size in height_choices if size <= 10]
        width_choices = [size for size in width_choices if size <= 10]
    elif frame_count == 2:
        height_choices = [size for size in height_choices if size <= 12]
        width_choices = [size for size in width_choices if size <= 12]

    if not height_choices or not width_choices:
        return None

    sizes = [(rng.choice(height_choices), rng.choice(width_choices)) for _ in range(frame_count)]
    sizes.sort(key=lambda item: item[0] * item[1], reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []

    for outer_h, outer_w in sizes:
        placed = False
        r_lo, r_hi = RESERVE_PAD, h - outer_h - RESERVE_PAD
        c_lo, c_hi = RESERVE_PAD, w - outer_w - RESERVE_PAD
        if r_lo > r_hi or c_lo > c_hi:
            return None
        for _ in range(FRAME_PLACEMENT_TRIES):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - RESERVE_PAD)
            rr1 = min(h, r1 + RESERVE_PAD + 1)
            cc0 = max(0, c0 - RESERVE_PAD)
            cc1 = min(w, c1 + RESERVE_PAD + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            frames.append((r0, r1, c0, c1))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    for r0, r1, c0, c1 in frames:
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

    marker_count = 0
    for bbox in frames:
        marker_count += _paint_quadrant_markers(grid, bbox, rng)

    if marker_count == 0:
        return None

    if _detect_frames(grid) != sorted(frames):
        return None

    out = main(grid)
    if not np.any((out != BG) & (out != FRAME)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid sample")
